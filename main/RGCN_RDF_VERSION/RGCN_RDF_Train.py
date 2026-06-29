import torch
import torch.nn as nn
import pandas as pd
from rdflib import Graph, Literal
from rdflib.namespace import RDF
from torch_geometric.data import Data
from torch_geometric.nn import FastRGCNConv
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0) 

# ==========================================
# 1. Configuration & Setup
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

ignore_relations = {
    'http://www.w3.org/2000/01/rdf-schema#domain',
    'http://www.w3.org/2000/01/rdf-schema#range',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf',
    'http://www.w3.org/2002/07/owl#disjointWith',
    'http://dl-learner.org/carcinogenesis#amesTestPositive'
}

# ==========================================
# 2. Load Datasets
# ==========================================
print("Loading RDF and TSV files...")
graph = Graph()
graph.parse('../mutag-hetero/mutag_stripped.nt', format="nt")

train_df = pd.read_csv('../mutag-hetero/trainingSet.tsv', sep='\t')
test_df = pd.read_csv('../mutag-hetero/testSet.tsv', sep='\t')

# ==========================================
# 3. Data Parsing & ID Mapping
# ==========================================
print("Parsing graph and extracting nodes/relations...")
nodes = set()
relations = set()
valid_triples = []
node_types = {} 

for s, p, o in graph:
    p_str = str(p) 

    if p == RDF.type and not isinstance(o, Literal):
        node_types[str(s)] = str(o).split("#")[-1]

    if p_str in ignore_relations or isinstance(o, Literal):
        continue

    s_str, o_str = str(s), str(o)
    
    nodes.add(s_str)
    nodes.add(o_str)
    relations.add(p_str)
    relations.add(p_str + "_inverse")
    valid_triples.append((s_str, p_str, o_str))

nodes = sorted(nodes)
relations = sorted(relations)

node_to_id = {node: i for i, node in enumerate(nodes)}
relation_to_id = {rel: i for i, rel in enumerate(relations)}

print(f"Filtered Unique Nodes: {len(node_to_id)}")
print(f"Total Relations (Including Inverses): {len(relation_to_id)}")

# ==========================================
# 4. Building PyG Tensors (Edges & Features)
# ==========================================
print("Building edge indices and feature matrix...")

src_list, dst_list, edge_type_list = [], [], []
for s_str, p_str, o_str in valid_triples:
    src_id, dst_id = node_to_id[s_str], node_to_id[o_str]
    rel_id = relation_to_id[p_str]
    inv_rel_id = relation_to_id[p_str + "_inverse"]
    
    src_list.append(src_id)
    dst_list.append(dst_id)
    edge_type_list.append(rel_id)
    
    src_list.append(dst_id)
    dst_list.append(src_id)
    edge_type_list.append(inv_rel_id)

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
edge_type = torch.tensor(edge_type_list, dtype=torch.long)

# Build Feature Matrix (One-Hot Encoding)
classes = sorted(list(set(node_types.values())))
class_to_id = {cls: i for i, cls in enumerate(classes)}
num_node_features = len(classes)

x = torch.zeros((len(node_to_id), num_node_features), dtype=torch.float)
for node_str, node_id in node_to_id.items():
    if node_str in node_types:
        cls_str = node_types[node_str]
        cls_id = class_to_id[cls_str]
        x[node_id, cls_id] = 1.0  

# ==========================================
# 5. Train / Test Masks
# ==========================================
def get_node_indices_and_labels(df, node_mapping):
    indices, labels = [], []
    for _, row in df.iterrows():
        uri = str(row['bond']).strip()
        if uri in node_mapping:
            indices.append(node_mapping[uri])
            labels.append(int(row['label_mutagenic']))
    return torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

train_idx, train_y = get_node_indices_and_labels(train_df, node_to_id)
test_idx, test_y = get_node_indices_and_labels(test_df, node_to_id)

data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
data.train_idx, data.train_y = train_idx, train_y
data.test_idx, data.test_y = test_idx, test_y

print(f"Data object ready. Nodes: {data.num_nodes}, Features: {data.x.shape[1]}")

# ==========================================
# 6. Model Definition
# ==========================================
class RGCN(nn.Module):
    def __init__(self, in_channels, num_rels, num_classes, hidden_dim=16):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_rels)
        self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_rels)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        return self.classifier(x)

model = RGCN(in_channels=data.x.shape[1], num_rels=len(relation_to_id), num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# ==========================================
# 7. Training Loop
# ==========================================
def train(data, device):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x, data.edge_index, data.edge_type)
    loss = criterion(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(data, device):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index, data.edge_type)
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_idx] == data.train_y).sum().item() / len(data.train_idx)
    test_acc = (pred[data.test_idx] == data.test_y).sum().item() / len(data.test_idx)
    return train_acc, test_acc

if __name__ == "__main__":
    print("\nStarting training on RDF graph...")
    for epoch in range(1, 71):
        loss = train(data, device)
        if epoch % 10 == 0:
            train_acc, test_acc = test(data, device)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print("Training finished!")

    print("Saving model and metadata for explainability pipeline...")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        
        'hyperparameters': {
            'in_channels': data.x.shape[1],  # 113
            'num_rels': len(relation_to_id), # 10
            'num_classes': 2,
            'hidden_dim': 16
        },
        
        'mappings': {
            'node_to_id': node_to_id,
            'relation_to_id': relation_to_id,
            'class_to_id': class_to_id
        },
        
        'graph_data': data.cpu()
    }

    save_path = 'rgcn_rdf_model_checkpoint.pt'
    torch.save(checkpoint, save_path)

    print(f"Successfully saved to {save_path}")