# XAI-MUTAG: Explainable AI for Mutagenicity Prediction

This project applies Relational Graph Convolutional Networks (RGCN) to predict mutagenicity of molecules from the MUTAG dataset, and uses GNNExplainer to generate post-hoc explanations for the model's predictions.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
- [Step 1: Dataset Preprocessing](#step-1-dataset-preprocessing)
- [Step 2: Training the RGCN Model](#step-2-training-the-rgcn-model)
- [Step 3: Generating Explanations](#step-3-generating-explanations)
- [Step 4: K-Fold Cross-Validation](#step-4-k-fold-cross-validation)
- [Step 5: Evaluating the Explainer](#step-5-evaluating-the-explainer)
- [Visualizing Explanations](#visualizing-explanations)
- [Output Files Reference](#output-files-reference)

---

## Project Structure

```
xai-mutag/
└── main/
    ├── process_dataset.py          # Step 1: Preprocesses raw RDF data into PyG dataset
    ├── requirements.txt            # Python dependencies
    ├── visualizer.html             # Standalone HTML visualizer for explanation payloads
    ├── mutag-hetero/               # Dataset directory (input + generated artifacts)
    │   ├── mutag_stripped.nt       # Raw RDF triples (source knowledge graph)
    │   ├── trainingSet.tsv         # Training split (80%)
    │   ├── testSet.tsv             # Test split (20%)
    │   ├── completeDataset.tsv     # Full dataset labels
    │   └── pyg_dataset.pt          # [GENERATED] Processed PyTorch Geometric dataset
    ├── models/                     # Saved model checkpoints
    │   └── rgnn_model_checkpoint.pt  # [GENERATED] Trained RGCN model
    └── RGCN/                       # RGCN pipeline scripts
        ├── RGCN_Train.py           # Step 2: Train the RGCN model
        ├── RGCN_GNNExplainer.py    # Step 3: Generate explanations for a molecule
        ├── RGCN_Kfold.py           # Step 4: 10-fold stratified cross-validation
        ├── evaluate_explainer.py   # Step 5: Evaluate explainer quality metrics
        └── payload.json            # [GENERATED] Explanation output for a molecule
```

---

## Prerequisites

- **Python** 3.12.13
- **Conda** (recommended for environment management)
- **CUDA** (optional, for GPU acceleration)

---

## Setup

### 1. Create and activate a Conda environment

```bash
conda create -n xai-env python=3.12 -y
conda activate xai-env
```

### 2. Install dependencies

From the `main/` directory:

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` contains many system-level packages. If you encounter conflicts, the core dependencies you actually need are:
> - `torch` (PyTorch)
> - `torch-geometric` (PyG)
> - `rdflib`
> - `scikit-learn`
> - `pandas`
> - `numpy`

### 3. Verify the dataset is in place

Ensure the `mutag-hetero/` directory in `main/` contains these files:

| File                  | Description                          |
|-----------------------|--------------------------------------|
| `mutag_stripped.nt`   | RDF knowledge graph (N-Triples)     |
| `trainingSet.tsv`     | Training set molecule URIs + labels  |
| `testSet.tsv`         | Test set molecule URIs + labels      |
| `completeDataset.tsv` | Full dataset with all molecule URIs  |

---

## Pipeline Overview

The pipeline must be followed in order, as each step depends on outputs from the previous step:

```
process_dataset.py  →  RGCN_Train.py  →  RGCN_GNNExplainer.py
                          ↓                        ↓
                     RGCN_Kfold.py         evaluate_explainer.py
```

**All scripts must be run from the `main/` directory** (i.e., your working directory should be `xai-mutag/main/`).

---

## Step 1: Dataset Preprocessing

This script parses the raw RDF knowledge graph and converts it into a PyTorch Geometric (`pyg_dataset.pt`) dataset that all downstream scripts use.

```bash
cd path/to/xai-mutag/main
python process_dataset.py
```

**What it does:**
1. Loads the RDF graph from `mutag-hetero/mutag_stripped.nt`
2. Reads training/test split labels from the TSV files
3. Extracts each molecule's atoms (node types) and bonds (edge types + endpoints)
4. One-hot encodes atom types as node features
5. Encodes bond types as integer edge types
6. Saves the processed dataset as `mutag-hetero/pyg_dataset.pt`

**Output:** `mutag-hetero/pyg_dataset.pt` — a list of `torch_geometric.data.Data` objects, each representing one molecule graph with:
- `x` — one-hot atom type features
- `edge_index` — bidirectional edge connectivity
- `edge_type` — integer-encoded bond type per edge
- `y` — mutagenicity label (0 or 1)
- `molecule_id` — molecule identifier (e.g. `"d305"`)
- `atom_info` — list of atom metadata (URI and type)
- `bond_info` — list of bond metadata (source idx, target idx, bond type string)

---

## Step 2: Training the RGCN Model

Train a 2-layer FastRGCN on the preprocessed dataset.

```bash
cd path/to/xai-mutag/main
python -m RGCN.RGCN_Train
```

> **Important:** You must run this as a module (`-m RGCN.RGCN_Train`) from the `main/` directory so that the relative imports in downstream scripts resolve correctly.

**Hyperparameters (defaults in the script):**

| Parameter         | Value  |
|-------------------|--------|
| Hidden channels   | 64     |
| Epochs            | 50     |
| Learning rate     | 0.001  |
| Batch size        | 32     |
| Optimizer         | Adam   |
| Loss function     | CrossEntropyLoss |

**What it does:**
1. Loads `pyg_dataset.pt` and the train/test TSV splits
2. Further splits training data into 80% train / 20% validation
3. Trains the RGCN model and prints train/val accuracy every 5 epochs
4. Reports final test accuracy
5. Saves the model checkpoint to `models/rgnn_model_checkpoint.pt`

**Output:** `models/rgnn_model_checkpoint.pt` containing:
- `model_state_dict` — trained model weights
- `optimizer_state_dict` — optimizer state
- `hyperparameters` — `in_channels`, `num_relations`, `hidden_channels`, `num_classes`

---

## Step 3: Generating Explanations

Use GNNExplainer to explain which atoms and bonds the model considers important for a given molecule's prediction.

```bash
cd path/to/xai-mutag/main
python -m RGCN.RGCN_GNNExplainer --uri "http://dl-learner.org/carcinogenesis#d305"
```

**Arguments:**

| Argument   | Required | Default                                                   | Description                              |
|------------|----------|-----------------------------------------------------------|------------------------------------------|
| `--uri`    | No       | `http://dl-learner.org/carcinogenesis#d101`               | Full URI of the molecule to explain      |

**Example — explain molecule d305:**
```bash
python -m RGCN.RGCN_GNNExplainer --uri "http://dl-learner.org/carcinogenesis#d305"
```

**Example — explain molecule d50:**
```bash
python -m RGCN.RGCN_GNNExplainer --uri "http://dl-learner.org/carcinogenesis#d50"
```

**What it does:**
1. Loads the trained model, PyG dataset, and RDF graph
2. Runs a forward pass to get the model's prediction and confidence
3. Applies GNNExplainer (200 epochs, lr=0.01) to compute node and edge importance masks
4. Normalizes importance scores to \[0, 1]
5. Saves the explanation as `RGCN/payload.json`

**Output:** `RGCN/payload.json` — a JSON file with the following structure:

```json
{
  "metadata": {
    "molecule_id": "d305",
    "uri": "http://dl-learner.org/carcinogenesis#d305",
    "true_class": 0
  },
  "prediction": {
    "predicted_class": 0,
    "confidence": 0.8491
  },
  "graph": {
    "nodes": [
      { "id": 0, "uri": "d305_1", "element": "Carbon", "importance_score": 0.1112 }
    ],
    "links": [
      { "source": 0, "target": 11, "type": "Bond-7", "importance_score": 0.1488 }
    ]
  }
}
```

Each node's `importance_score` reflects how much that atom contributed to the prediction (1.0 = most important). Each edge's `importance_score` reflects bond importance (averaged over both directions).

---

## Step 4: K-Fold Cross-Validation

Run 10-fold stratified cross-validation to get a robust estimate of model performance.

```bash
cd path/to/xai-mutag/main
python -m RGCN.RGCN_Kfold
```

**What it does:**
1. Uses the **full** dataset (train + test combined, ~340 molecules)
2. Applies `StratifiedKFold` with 10 folds to preserve the class ratio
3. Trains a fresh model on each fold (same hyperparameters as Step 2)
4. Reports per-fold test accuracy and summary statistics (mean, std, 95% CI)

**Output:** Printed to console only (results are not saved to file).

---

## Step 5: Evaluating the Explainer

Compute explanation quality metrics across all correctly classified test molecules.

```bash
cd path/to/xai-mutag/main
python -m RGCN.evaluate_explainer
```

> **Important:** Make sure the model checkpoint exists in `models/` before running this.

**What it does:**
1. Loads the trained model and the test set
2. For each correctly classified test molecule, generates an explanation and computes:
   - **Fidelity+** — drop in predicted-class probability when the important subgraph is removed (higher = important subgraph truly drives the prediction)
   - **Fidelity-** — drop in predicted-class probability when keeping only the important subgraph (higher = important subgraph alone is sufficient)
   - **Unfaithfulness** — how well the explanation's mask approximates the model's behavior (lower is better)
   - **Sparsity** — fraction of the graph not in the top-k important elements (higher = more concise explanation)
3. Reports per-molecule metrics and overall averages
4. Saves results to `RGCN/explanation_evaluation_results.pt`

**Output:** `RGCN/explanation_evaluation_results.pt` containing:
- `per_molecule_results` — list of dicts with metrics per molecule
- `averages` — mean of each metric across all correctly classified molecules
- `num_correctly_classified` — count of test molecules the model predicted correctly
- `total_test` — total test set size

---

## Visualizing Explanations

A standalone HTML visualizer is included at `main/visualizer.html`. It uses D3.js to render the explanation graph interactively.

### How to use

1. Generate an explanation payload (Step 3) — this creates `RGCN/payload.json`
2. Open `visualizer.html` in any modern web browser
3. Upload the `payload.json` file via the drag-and-drop upload zone
4. Interact with the visualization:
   - Hover over nodes/edges to see detailed importance scores
   - Adjust node size, edge width, and link distance with sliders
   - Switch between 4 color themes (purple, cyan, emerald, orange)
   - Toggle node/edge labels on/off
   - Zoom and pan the graph

---

## Output Files Reference

| File                                         | Generated by              | Description                                    |
|----------------------------------------------|---------------------------|------------------------------------------------|
| `mutag-hetero/pyg_dataset.pt`                | `process_dataset.py`      | Preprocessed PyG graph dataset                 |
| `models/rgnn_model_checkpoint.pt`            | `RGCN/RGCN_Train.py`      | Trained RGCN model checkpoint                  |
| `RGCN/payload.json`                          | `RGCN/RGCN_GNNExplainer.py` | Explanation payload for a single molecule     |
| `RGCN/explanation_evaluation_results.pt`      | `RGCN/evaluate_explainer.py` | Aggregate explainer quality metrics          |
