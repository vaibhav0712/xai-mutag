from torch_geometric.nn import global_add_pool, PNAConv
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
import os 

ROOT_DIR = Path().cwd().parent
models_dir = ROOT_DIR / "models"
dataset_path = ROOT_DIR / "mutag-hetero"

# check if path exist 
if not models_dir.exists():
    raise FileNotFoundError(f"Models directory path {models_dir} does not exist.")


def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(0)

class PNA_Molecule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_edge_types,
        num_classes,
        deg # <-- REQUIRED FOR PNA
    ):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_channels)

        # Define the multiple aggregators and scalers that make PNA special
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # PNA Layers (Notice we pass the aggregators, scalers, and deg tensor)
        self.conv1 = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=hidden_channels, towers=1, pre_layers=1, post_layers=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=hidden_channels, towers=1, pre_layers=1, post_layers=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=hidden_channels, towers=1, pre_layers=1, post_layers=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(0.5)

        # Classifier using Jumping Knowledge (Concat of 3 layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x, edge_index, edge_type, batch):
        x = self.node_encoder(x.float())
        edge_attr = self.edge_embedding(edge_type)

        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr=edge_attr)))
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr=edge_attr)))
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index, edge_attr=edge_attr)))
        
        x_jk = torch.cat([x1, x2, x3], dim=-1)
        pool_x = global_add_pool(x_jk, batch)
        
        return self.classifier(pool_x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.batch)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.squeeze()).sum().item()
    return correct / len(loader.dataset)


BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print(f"[INFO] Training on device: {device}")

print("[INFO] Loading processed dataset...")
pyg_dataset = torch.load(dataset_path / "pyg_dataset.pt", weights_only=False)    
train_df = pd.read_csv(dataset_path / "trainingSet.tsv", sep="\t")
test_df = pd.read_csv(dataset_path / "testSet.tsv", sep="\t")


print("[INFO] Mapping training and testing molecule IDs to dataset indices...")
id_to_idx = {data.molecule_id: i for i, data in enumerate(pyg_dataset)}
  
train_ids = sorted(train_df["bond"].apply(lambda x: x.split("#")[-1]))
test_ids = sorted(test_df["bond"].apply(lambda x: x.split("#")[-1]))

train_idx = [id_to_idx[mol_id] for mol_id in train_ids if mol_id in id_to_idx]
test_idx  = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

print("[INFO] Constructing train, test and validation DataLoader")
train_loader = DataLoader([pyg_dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader([pyg_dataset[i] for i in val_idx], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader([pyg_dataset[i] for i in test_idx], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

in_channels = pyg_dataset[0].x.size(1)
num_relations = int(max(data.edge_type.max().item() for data in pyg_dataset) + 1)
num_edge_types = num_relations

print("[INFO] Computing degree distribution for PNA...")
max_degree = -1
for data in pyg_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))
deg = torch.zeros(max_degree + 1, dtype=torch.long)
for data in pyg_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
deg = deg.to(torch.float)



def get_fresh_model(name):
    if name == 'pna':
        return PNA_Molecule(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            num_edge_types=num_edge_types,
            num_classes=2,
            deg=deg # <-- Pass the degree tensor here!
        ).to(device)
    
BATCH_SIZE = 16
HIDDEN_CHANNELS = 16
EPOCHS = 50
LR = 0.001
#######################################################
if __name__ == "__main__":
    print("[INFO] Building PNA model...")
    model = get_fresh_model("pna")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=5e-4
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("[INFO] Starting training...")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

    print(f"\n[INFO] Best Validation Accuracy: {best_val_acc:.4f}")

    # Load best validation model before testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_acc = evaluate(model, test_loader, device)

    print(f"[INFO] Final Test Accuracy: {test_acc:.4f}")

    print("[INFO] Saving model checkpoint...")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": {
            "in_channels": in_channels,
            "hidden_channels": HIDDEN_CHANNELS,
            "num_edge_types": num_edge_types,
            "num_classes": 2,
            "deg": deg
        }
    }

    torch.save(
        checkpoint,
        models_dir / "pna_model_checkpoint.pt"
    )

    print("[INFO] Saved successfully as 'pna_model_checkpoint.pt'")
