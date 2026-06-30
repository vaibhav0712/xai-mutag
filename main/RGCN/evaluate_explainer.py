import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import unfaithfulness
from torch_geometric.nn import FastRGCNConv, global_mean_pool

from RGCN.RGCN_Train import MutagenicityGNN, set_seed
from RGCN.RGCN_GNNExplainer import RGCNExplainerWrapper


ROOT_DIR = Path().cwd()
MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "mutag-hetero"

MODEL_PATH = MODELS_DIR / "rgnn_model_checkpoint.pt"
DATASET_PATH = DATASET_DIR / "pyg_dataset.pt"
TRAIN_TSV = DATASET_DIR / "trainingSet.tsv"
TEST_TSV = DATASET_DIR / "testSet.tsv"

SEED = 42


# custom sparsity function
def compute_sparsity(explanation, data, topk=10):
    """
    Sparsity = fraction of graph elements NOT in the top-k important subgraph.
    Normalised per-graph so varying graph sizes are handled fairly.

    For each graph we:
      1. Take the top-k most important edges by raw edge_mask score.
      2. Count nodes incident to those top-k edges.
      3. Sparsity = 1 - (selected_nodes + selected_edges) / (total_nodes + total_edges)

    Higher sparsity = more concise explanation (fewer elements used).
    """
    edge_mask = explanation.edge_mask
    num_edges = edge_mask.numel()
    num_nodes = data.num_nodes

    if num_edges == 0:
        return 1.0

    k = min(topk, num_edges)
    _, top_edge_indices = edge_mask.topk(k)
    selected_edges = torch.zeros(num_edges, device=edge_mask.device)
    selected_edges[top_edge_indices] = 1.0

    src = data.edge_index[0]
    dst = data.edge_index[1]
    incident_nodes = set()
    for ei in top_edge_indices.tolist():
        incident_nodes.add(src[ei].item())
        incident_nodes.add(dst[ei].item())

    num_selected_edges = int(selected_edges.sum().item())
    num_selected_nodes = len(incident_nodes)

    total_elements = num_nodes + num_edges
    selected_elements = num_selected_nodes + num_selected_edges

    sparsity = 1.0 - (selected_elements / total_elements)
    return sparsity

# custom fiedlity function. reason for using this is mentioned in report.
def compute_fidelity(explainer, explanation, data, device):
    """
    Compute probability-based fidelity+ and fidelity- for graph-level
    classification.

    Instead of the binary "did the predicted class change?", we measure the
    drop (or rise) in the softmax probability of the original predicted class.

    fidelity+  = P(orig) - P(complement)   [remove important subgraph]
      High => the important subgraph was truly carrying the prediction.
    fidelity-  = P(orig) - P(subgraph)     [keep only important subgraph]
      High => the important subgraph alone is sufficient for the prediction.

    Both values are in [0, 1]. 0 = no change, 1 = probability dropped to 0.
    """
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        original_logits = explainer.model(
            data.x, data.edge_index, data.edge_type, batch
        )
        original_probs = F.softmax(original_logits, dim=-1).squeeze()
        pred_class = original_logits.argmax(dim=-1).item()
        original_prob = original_probs[pred_class].item()

    with torch.no_grad():
        if node_mask is not None:
            x_complement = (1.0 - node_mask) * data.x
        else:
            x_complement = data.x

        if edge_mask is not None:
            complement_mask = 1.0 - edge_mask
            keep = complement_mask > 0.5
            masked_edge_index = data.edge_index[:, keep]
            masked_edge_type = data.edge_type[keep]
        else:
            masked_edge_index = data.edge_index
            masked_edge_type = data.edge_type

        if masked_edge_index.size(1) == 0:
            fidelity_pos = original_prob
        else:
            batch_c = torch.zeros(x_complement.size(0), dtype=torch.long, device=device)
            complement_logits = explainer.model(
                x_complement, masked_edge_index, masked_edge_type, batch_c
            )
            complement_probs = F.softmax(complement_logits, dim=-1).squeeze()
            complement_prob = complement_probs[pred_class].item()
            fidelity_pos = max(original_prob - complement_prob, 0.0)

    with torch.no_grad():
        if node_mask is not None:
            x_subgraph = node_mask * data.x
        else:
            x_subgraph = data.x

        if edge_mask is not None:
            keep = edge_mask > 0.5
            masked_edge_index = data.edge_index[:, keep]
            masked_edge_type = data.edge_type[keep]
        else:
            masked_edge_index = data.edge_index
            masked_edge_type = data.edge_type

        if masked_edge_index.size(1) == 0:
            fidelity_neg = original_prob
        else:
            batch_s = torch.zeros(x_subgraph.size(0), dtype=torch.long, device=device)
            subgraph_logits = explainer.model(
                x_subgraph, masked_edge_index, masked_edge_type, batch_s
            )
            subgraph_probs = F.softmax(subgraph_logits, dim=-1).squeeze()
            subgraph_prob = subgraph_probs[pred_class].item()
            fidelity_neg = max(original_prob - subgraph_prob, 0.0)

    return fidelity_pos, fidelity_neg


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading dataset...")
    pyg_dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"[INFO] Total molecules in dataset: {len(pyg_dataset)}")

    print("[INFO] Loading train/test split...")
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    test_df = pd.read_csv(TEST_TSV, sep="\t")

    id_to_idx = {data.molecule_id: i for i, data in enumerate(pyg_dataset)}
    test_ids = sorted(test_df["bond"].apply(lambda x: x.split("#")[-1]))
    test_idx = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]
    print(f"[INFO] Test set size: {len(test_idx)}")

    print("[INFO] Loading model checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    hparams = checkpoint["hyperparameters"]

    base_model = MutagenicityGNN(
        in_channels=hparams["in_channels"],
        num_relations=hparams["num_relations"],
        hidden_channels=hparams["hidden_channels"],
        num_classes=hparams["num_classes"],
    ).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    wrapper_model = RGCNExplainerWrapper(base_model)

    print("[INFO] Initializing GNNExplainer...")
    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    print("[INFO] Explainer ready.")

    results = []

    correctly_classified = 0
    total_test = len(test_idx)

    print(f"[INFO] Evaluating explanations for {total_test} test molecules...")
    print("=" * 60)

    for i, idx in enumerate(test_idx):
        data = pyg_dataset[idx].to(device)
        mol_id = data.molecule_id
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            logits = wrapper_model(data.x, data.edge_index, data.edge_type, batch)
            pred_class = logits.argmax(dim=1).item()
            true_class = data.y.item()

        is_correct = pred_class == true_class

        if not is_correct:
            print(f"[{i+1}/{total_test}] mol={mol_id}  SKIP (misclassified: pred={pred_class} true={true_class})")
            continue

        correctly_classified += 1
        print(f"[{i+1}/{total_test}] mol={mol_id}  pred={pred_class} true={true_class}  -> generating explanation...")

        set_seed(SEED)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            batch=batch,
        )

        print(f"[INFO]   Computing metrics...")
        unfaith_score = unfaithfulness(explainer=explainer, explanation=explanation)
        fid_pos, fid_neg = compute_fidelity(explainer, explanation, data, device)
        sparsity_score = compute_sparsity(explanation, data, topk=10)

        result = {
            "molecule_id": mol_id,
            "predicted_class": pred_class,
            "true_class": true_class,
            "fidelity_pos": fid_pos,
            "fidelity_neg": fid_neg,
            "unfaithfulness": unfaith_score,
            "sparsity": sparsity_score,
        }
        results.append(result)

        print(f"    fidelity_pos={fid_pos:.4f}  fidelity_neg={fid_neg:.4f}  "
              f"unfaithfulness={unfaith_score:.4f}  sparsity={sparsity_score:.4f}")

    print("=" * 60)
    print(f"[INFO] Correctly classified: {correctly_classified} / {total_test}")

    if len(results) == 0:
        print("[WARNING] No correctly classified molecules found. No averages to report.")
    else:
        avg_fid_pos = np.mean([r["fidelity_pos"] for r in results])
        avg_fid_neg = np.mean([r["fidelity_neg"] for r in results])
        avg_unfaith = np.mean([r["unfaithfulness"] for r in results])
        avg_sparsity = np.mean([r["sparsity"] for r in results])

        print(f"\n{'='*60}")
        print(f"  Average Explanation Metrics ({len(results)} correctly classified)")
        print(f"{'='*60}")
        print(f"  Avg Fidelity (+)     : {avg_fid_pos:.4f}")
        print(f"  Avg Fidelity (-)      : {avg_fid_neg:.4f}")
        print(f"  Avg Unfaithfulness    : {avg_unfaith:.4f}  (lower is better)")
        print(f"  Avg Sparsity          : {avg_sparsity:.4f}  (higher is better)")
        print(f"{'='*60}")

        output_path = Path(__file__).parent / "explanation_evaluation_results.pt"
        torch.save(
            {
                "per_molecule_results": results,
                "averages": {
                    "fidelity_pos": avg_fid_pos,
                    "fidelity_neg": avg_fid_neg,
                    "unfaithfulness": avg_unfaith,
                    "sparsity": avg_sparsity,
                },
                "num_correctly_classified": len(results),
                "total_test": total_test,
                "seed": SEED,
            },
            output_path,
        )
        print(f"[INFO] Results saved to {output_path}")
