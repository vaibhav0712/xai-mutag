from train import MutagenicityGNN
import torch.nn.functional as F
from rdflib import Graph
from torch_geometric.explain import Explainer, GNNExplainer
import torch

class RGCNExplainerWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_type, batch):
        for module in self.base_model.modules():
            if hasattr(module, 'explain'):
                module.explain = False
        
        return self.base_model(x, edge_index, edge_type, batch)

def short_name(uri):
    """Extracts the human-readable string from an RDF URI."""
    return str(uri).split("#")[-1]

def initialize_xai_pipeline(model_checkpoint_path, dataset_path, rdf_path):
    """
    Loads all heavy assets into memory once. 
    Returns a dictionary of resources to be injected into your endpoints.
    """
    print("[INFO] Initializing XAI Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading PyG dataset...")
    pyg_dataset = torch.load(dataset_path, weights_only=False)

    unique_bonds = set()
    for data in pyg_dataset:
        for _, _, bond_str in data.bond_info:
            unique_bonds.add(bond_str)
    bond_encoder = {b: i for i, b in enumerate(sorted(unique_bonds))}
    bond_decoder = {i: b for b, i in bond_encoder.items()}

    # 3. Load Model
    print("[INFO] Loading model checkpoint...")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    hparams = checkpoint['hyperparameters']
    
    base_model = MutagenicityGNN(
        in_channels=hparams['in_channels'],
        num_relations=hparams['num_relations'],
        hidden_channels=hparams['hidden_channels'],
        num_classes=hparams['num_classes']
    ).to(device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    # Wrap for explainer to prevent RGCN crash
    wrapper_model = RGCNExplainerWrapper(base_model)

    print("[INFO] Initializing GNN Explainer...")
    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )

    print("[INFO] Parsing RDF Knowledge Graph...")
    g = Graph()
    g.parse(rdf_path, format="nt")
    
    print("[INFO] Pipeline ready.")
    
    return {
        "model": wrapper_model,
        "explainer": explainer,
        "pyg_dataset": pyg_dataset,
        "rdf_graph": g,
        "bond_decoder": bond_decoder,
        "device": device
    }

def generate_explanation_payload(molecule_uri_str, pipeline_resources):
    """
    Takes a molecule URI and the cached pipeline resources, returning JSON.
    """
    # Unpack resources
    model = pipeline_resources["model"]
    explainer = pipeline_resources["explainer"]
    pyg_dataset = pipeline_resources["pyg_dataset"]
    g = pipeline_resources["rdf_graph"]
    bond_decoder = pipeline_resources["bond_decoder"]
    device = pipeline_resources["device"]

    molecule_id = molecule_uri_str.split('#')[-1]
    
    data_idx = next((i for i, d in enumerate(pyg_dataset) if d.molecule_id == molecule_id), None)
    if data_idx is None:
        return json.dumps({"error": f"Molecule {molecule_id} not found."})
    
    data = pyg_dataset[data_idx].to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    
    # Run Prediction
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_type, batch)
        probabilities = F.softmax(logits, dim=1).squeeze()
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
        
    # Generate Explanation
    explanation = explainer(
        x=data.x, edge_index=data.edge_index, edge_type=data.edge_type, batch=batch
    )
    
    if explanation.node_mask is not None:
        raw_scores = explanation.node_mask.sum(dim=1)
        max_score = raw_scores.max().item() if raw_scores.max().item() > 0 else 1.0
        node_scores = (raw_scores / max_score).cpu().tolist()
    else:
        node_scores = [0.0] * data.num_nodes

    # Construct Topology
    nodes_payload = []
    for i in range(data.num_nodes):
        nodes_payload.append({
            "id": i,
            "uri": data.atom_info[i]['uri'],
            "element": data.atom_info[i]['type'].split('-')[0],
            "importance_score": round(node_scores[i], 4)
        })
        
    edges_payload = []
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        if src < dst: 
            edges_payload.append({
                "source": src,
                "target": dst,
                "type": bond_decoder[data.edge_type[i].item()] # Fixed decoder usage
            })
    
    return json.dumps({
        "metadata": {"molecule_id": molecule_id, "uri": molecule_uri_str, "true_class": data.y.item()},
        "prediction": {"predicted_class": predicted_class, "confidence": round(confidence, 4)},
        "graph": {"nodes": nodes_payload, "links": edges_payload},
    })

if __name__ == "__main__":
    import time
    from pathlib import Path
    import json

    # Define paths based on your project structure
    ROOT_DIR = Path().cwd()
    models_dir = ROOT_DIR / "models"
    dataset_path = ROOT_DIR / "mutag-hetero"
    MODEL_PATH = models_dir / 'gnn_model_checkpoint.pt'
    DATASET_PATH = dataset_path / 'pyg_dataset.pt'
    RDF_PATH = dataset_path / 'mutag_stripped.nt'

    # Test Molecule URI (Using the d305 example you mentioned earlier)
    TEST_URI = "http://dl-learner.org/carcinogenesis#d76"

    print("=== Step 1: Testing Pipeline Initialization ===")
    start_time = time.time()
    
    try:
        resources = initialize_xai_pipeline(MODEL_PATH, DATASET_PATH, RDF_PATH)
        init_time = time.time() - start_time
        print(f"✅ Pipeline initialized successfully in {init_time:.2f} seconds!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        exit(1)

    print("\n=== Step 2: Testing Payload Generation ===")
    start_time = time.time()
    
    try:
        json_response = generate_explanation_payload(TEST_URI, resources)

        gen_time = time.time() - start_time
        print(f"✅ Payload generated successfully in {gen_time:.4f} seconds!\n")

        # Parse the JSON string back to a dict just to print a clean summary for the terminal
        parsed_json = json.loads(json_response)
        with open('payload.json', 'w') as f:
            json.dump(parsed_json, f)
        
        print("--- Output Summary ---")
        print(f"Metadata: {json.dumps(parsed_json.get('metadata'), indent=2)}")
        print(f"Prediction: {json.dumps(parsed_json.get('prediction'), indent=2)}")
        print(f"Nodes Extracted: {len(parsed_json.get('graph', {}).get('nodes', []))}")
        print(f"Edges Extracted: {len(parsed_json.get('graph', {}).get('links', []))}")
        
    except Exception as e:
        print(f"❌ Failed to generate payload: {e}")
