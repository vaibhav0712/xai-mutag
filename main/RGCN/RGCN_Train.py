import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from torch_geometric.nn import FastRGCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch

ROOT_DIR = Path().cwd()
models_dir = ROOT_DIR / "models"
dataset_path = ROOT_DIR / "mutag-hetero"

if not models_dir.exists():
    models_dir.mkdir(parents=True, exist_ok=True)


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
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


set_seed(132) # 42,0 


# 1. Model Definition
class MutagenicityGNN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = FastRGCNConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# 2. Training and Evaluation Functions
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


# 3. Main Execution Block
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    EPOCHS = 50
    LR = 0.001
    
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

    # Dynamically extract dimensions from the loaded data
    in_channels = pyg_dataset[0].x.size(1)
    
    # Extract total number of unique edge types across the dataset
    num_relations = int(max(data.edge_type.max().item() for data in pyg_dataset) + 1)

    model = MutagenicityGNN(
        in_channels=in_channels,
        num_relations=num_relations,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    print("[INFO] Starting training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    test_acc = evaluate(model, test_loader, device)
    print(f'[INFO] Final Test Accuracy: {test_acc:.4f}')

    print("[INFO] Saving model checkpoint...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {
            'in_channels': in_channels,
            'num_relations': num_relations,
            'hidden_channels': HIDDEN_CHANNELS,
            'num_classes': 2
        }
    }


    torch.save(checkpoint, models_dir / 'rgnn_model_checkpoint.pt')
    print("[INFO] Saved successfully as 'rgnn_model_checkpoint.pt'")


