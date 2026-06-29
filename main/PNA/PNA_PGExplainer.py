import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_add_pool, PNAConv
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import os
import numpy as np
from PNA.PNA_Train import PNA_Molecule

ROOT_DIR = Path().cwd()
models_dir = ROOT_DIR / "models"
dataset_path = ROOT_DIR / "mutag-hetero"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_model_and_data(device):
    print("[INFO] Loading dataset...")
    pyg_dataset = torch.load(dataset_path / "pyg_dataset.pt", weights_only=False)
    
    print("[INFO] Loading model checkpoint...")
    checkpoint_path = models_dir / "pna_model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hyperparameters"]
    
    model = PNA_Molecule(
        in_channels=hparams["in_channels"],
        hidden_channels=hparams["hidden_channels"],
        num_edge_types=hparams["num_edge_types"],
        num_classes=hparams["num_classes"],
        deg=hparams["deg"].to(device)
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, pyg_dataset

def train_explainer(explainer, model, dataset, device, epochs=10):
    print("[INFO] Training PGExplainer globally...")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(epochs):
        loss_total = 0
        for data in loader:
            data = data.to(device)
            loss = explainer.algorithm.train(
                epoch, model, data.x, data.edge_index, target=data.y,
                edge_type=data.edge_type, batch=data.batch
            )
            loss_total += loss
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Explainer Epoch {epoch:02d} | Loss: {loss_total:.4f}")

def explain_molecule(molecule_uri, model, explainer, dataset, device):
    """
    Finds the molecule by URI, runs the explainer, and formats a JSON payload
    matching the exact requested frontend structure.
    """
    target_data = None
    for data in dataset:
        if data.molecule_id == molecule_uri:
            target_data = data
            break
            
    if target_data is None:
        raise ValueError(f"Molecule with URI '{molecule_uri}' not found in dataset.")
        
    target_data = target_data.to(device)
    
    # Get standard prediction & confidence
    with torch.no_grad():
        out = model(target_data.x, target_data.edge_index, target_data.edge_type, target_data.batch)
        probabilities = F.softmax(out, dim=1)[0]
        pred_class = out.argmax(dim=1).item()
        confidence = probabilities[pred_class].item()
        target_class = target_data.y.item()
    
    # Run Explainer
    print(f"[INFO] Running explanation for {molecule_uri}...")
    explanation = explainer(
        target_data.x, 
        target_data.edge_index, 
        target=target_data.y,
        edge_type=target_data.edge_type,
        batch=target_data.batch
    )
    
    edge_mask = explanation.edge_mask.cpu().detach().numpy()
    edge_index = target_data.edge_index.cpu().numpy()
    min_val = edge_mask.min()
    max_val = edge_mask.max()
    
    if max_val - min_val > 0:
        edge_mask = (edge_mask - min_val) / (max_val - min_val)
    else:
        edge_mask = np.zeros_like(edge_mask) # Fallback if all scores are identical
    # ------------------------------
    
    edge_index = target_data.edge_index.cpu().numpy()
    
    # Derive Node Importance (casting mask_val to float)
    num_nodes = target_data.num_nodes
    node_importance = torch.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        mask_val = float(edge_mask[i]) 
        node_importance[u] = max(node_importance[u].item(), mask_val)
        node_importance[v] = max(node_importance[v].item(), mask_val)
    
    # Build Frontend JSON
    mol_id = molecule_uri.split('#')[-1] if '#' in molecule_uri else molecule_uri
    
    # Map typical dataset features to elements (Adjust indices based on your specific dataset)
    ELEMENT_MAP = {0: "Carbon", 1: "Nitrogen", 2: "Oxygen", 3: "Fluorine", 
                   4: "Iodine", 5: "Chlorine", 6: "Bromine", 7: "Hydrogen"}
    
    nodes_list = []
    for i in range(num_nodes):
        # Assuming node features are one-hot encoded atomic types
        elem_idx = target_data.x[i].argmax().item() if target_data.x.dim() > 1 else 0
        element = ELEMENT_MAP.get(elem_idx, f"Unknown_{elem_idx}")
        
        nodes_list.append({
            "id": i,
            "uri": f"{mol_id}_{i+1}",
            "element": element,
            "importance_score": round(float(node_importance[i]), 4)
        })

    links_list = []
    for i in range(edge_index.shape[1]):
        links_list.append({
            "source": int(edge_index[0, i]),
            "target": int(edge_index[1, i]),
            "type": f"Bond-{int(target_data.edge_type[i].item())}",
            "importance_score": round(float(edge_mask[i]), 4) # Added per your request for edge importance
        })

    result = {
        "metadata": {
            "molecule_id": mol_id,
            "uri": molecule_uri,
            "true_class": target_class
        },
        "prediction": {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        },
        "graph": {
            "nodes": nodes_list,
            "links": links_list
        }
    }
    
    return json.dumps(result, indent=4)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    model, dataset = load_model_and_data(device)
    
    # FIXED: Added .to(device) to the PGExplainer algorithm
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=10, lr=0.003, edge_size=0.001,
        edge_ent=0.0001).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    train_explainer(explainer, model, dataset, device, epochs=10)
    
    example_uri = dataset[0].molecule_id 
    json_output = explain_molecule(example_uri, model, explainer, dataset, device)
    
    output_file = ROOT_DIR / f"PNA_PGExplainer_Payload.json"
    with open(output_file, "w") as f:
        f.write(json_output)
        
    print(f"\n[INFO] Explanation saved to {output_file}")
    print(json_output)