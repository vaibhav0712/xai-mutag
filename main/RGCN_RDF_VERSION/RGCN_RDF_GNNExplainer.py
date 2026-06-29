import torch
import torch.nn as nn
import json
import argparse
import os
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import FastRGCNConv
from torch_geometric.seed import seed_everything



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


# Helper function to strip long URIs down to readable strings
def make_readable(uri_string):
    return uri_string.split('#')[-1].split('/')[-1]

def main(target_uri, checkpoint_path='rgcn_rdf_model_checkpoint.pt', output_json='explanation.json', top_k_edges=20, seed=42):
    # Enable determinism
    seed_everything(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
    
    # ==========================================
    # 2. Load Checkpoint & Data
    # ==========================================
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = checkpoint['hyperparameters']
    
    model = RGCN(hp['in_channels'], hp['num_rels'], hp['num_classes'], hp['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    data = checkpoint['graph_data'].to(device)
    
    # Reverse mappings
    node_to_id = checkpoint['mappings']['node_to_id']
    id_to_node = {v: k for k, v in node_to_id.items()}
    
    relation_to_id = checkpoint['mappings']['relation_to_id']
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    class_to_id = checkpoint['mappings']['class_to_id']
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    if target_uri not in node_to_id:
        raise ValueError(f"Target URI '{target_uri}' not found in the graph's nodes.")
    
    target_idx = node_to_id[target_uri]
    
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_type)
        probs = torch.softmax(out[target_idx], dim=-1)
        prediction = probs.argmax(dim=-1).item()
        confidence = probs[prediction].item()
    print(f"[INFO] Model prediction for {target_uri}: Class {prediction} (confidence: {confidence:.4f})")

    # ==========================================
    # 3. Setup and Run GNNExplainer
    # ==========================================
    print("[INFO] Generating explanation...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes', 
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    
    explanation = explainer(data.x, data.edge_index, target=prediction, index=target_idx, edge_type=data.edge_type)
    
    # ==========================================
    # 4. Extract Explanatory Subgraph
    # ==========================================
    edge_mask = explanation.edge_mask.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    edge_type = data.edge_type.cpu().numpy()
    
    top_edge_indices = edge_mask.argsort()[-top_k_edges:][::-1]
    
    # 1. Collect all unique nodes in the explanation
    nodes_in_explanation = set([target_uri])
    for idx in top_edge_indices:
        src_id = edge_index[0, idx]
        dst_id = edge_index[1, idx]
        nodes_in_explanation.update([id_to_node[src_id], id_to_node[dst_id]])

    # 2. Sort nodes to assign local IDs
    sorted_nodes = sorted(list(nodes_in_explanation))
    node_to_local_id = {uri: idx for idx, uri in enumerate(sorted_nodes)}

    # 3. Build nodes payload
    node_mask = explanation.node_mask.cpu().numpy() if 'node_mask' in explanation and explanation.node_mask is not None else None
    nodes_payload = []
    for uri in sorted_nodes:
        n_id = node_to_id[uri]
        feature_vec = data.x[n_id]
        
        # Decode the one-hot feature vector back to a class string
        if feature_vec.sum() > 0:
            class_idx = feature_vec.argmax().item()
            raw_class_str = id_to_class[class_idx]
            node_label = make_readable(raw_class_str)
        else:
            node_label = "Unknown"
            
        node_importance = 0.0
        if node_mask is not None:
            mask_val = node_mask[n_id]
            node_importance = float(mask_val.sum()) if mask_val.ndim > 0 else float(mask_val)

        nodes_payload.append({
            "id": node_to_local_id[uri],
            "uri": uri,
            "element": node_label,
            "importance_score": round(node_importance, 4)
        })

    # 4. Build links payload
    links_payload = []
    for idx in top_edge_indices:
        weight = float(edge_mask[idx])
        src_id = edge_index[0, idx]
        dst_id = edge_index[1, idx]
        rel_id = edge_type[idx]
        
        src_uri = id_to_node[src_id]
        dst_uri = id_to_node[dst_id]
        rel_str = id_to_relation[rel_id]
        
        links_payload.append({
            "source": node_to_local_id[src_uri],
            "target": node_to_local_id[dst_uri],
            "importance_score": round(weight, 4),
            "type": make_readable(rel_str)
        })

    # Sort links_payload deterministically to break any weight ties consistently
    links_payload_sorted = sorted(
        links_payload,
        key=lambda e: (-e["importance_score"], e["source"], e["target"], e["type"])
    )

    # 5. Extract metadata (molecule_id, uri, true_class)
    molecule_id = make_readable(target_uri)
    
    true_class = None
    if hasattr(data, 'train_idx') and data.train_idx is not None and hasattr(data, 'train_y'):
        matching_indices = (data.train_idx == target_idx).nonzero(as_tuple=True)[0]
        if len(matching_indices) > 0:
            true_class = int(data.train_y[matching_indices[0]].item())
    if true_class is None and hasattr(data, 'test_idx') and data.test_idx is not None and hasattr(data, 'test_y'):
        matching_indices = (data.test_idx == target_idx).nonzero(as_tuple=True)[0]
        if len(matching_indices) > 0:
            true_class = int(data.test_y[matching_indices[0]].item())

    # ==========================================
    # 5. Build and Save JSON Payload
    # ==========================================
    payload = {
        "metadata": {
            "molecule_id": molecule_id,
            "uri": target_uri,
            "true_class": true_class
        },
        "prediction": {
            "predicted_class": prediction,
            "confidence": round(confidence, 4)
        },
        "graph": {
            "nodes": nodes_payload,
            "links": links_payload_sorted
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)
        
    print(f"[INFO] Explanation successfully saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNNExplainer on a trained RGCN model.")
    parser.add_argument('--uri', type=str, required=False, help="The URI of the node to explain.")
    parser.add_argument('--top_k', type=int, default=15, help="Number of top important edges to save.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    target_uri = args.uri if args.uri else 'http://dl-learner.org/carcinogenesis#d300'
    main(target_uri=target_uri, top_k_edges=args.top_k, seed=args.seed)