from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
import torch

from RGCN.RGCN_Train import MutagenicityGNN, train_epoch, evaluate, set_seed

ROOT_DIR    = Path().cwd()
DATASET_DIR = ROOT_DIR / "mutag-hetero"


def run_kfold(
    pyg_dataset,
    n_splits=10,
    hidden_channels=64,
    epochs=50,
    lr=0.001,
    batch_size=32,
    seed=42,
    device=None,
):
    """
    Stratified K-Fold cross-validation over the full PyG dataset.

    Uses StratifiedKFold (not plain KFold) so each fold preserves the
    mutagenic / non-mutagenic class ratio — important for the slight class
    imbalance in MUTAG.

    The full dataset is used (train + test molecules combined) because with
    ~340-380 molecules a fixed hold-out wastes too much signal.  Every molecule
    appears in exactly one test fold across the 10 folds.

    Parameters
    ----------
    pyg_dataset      : list[Data]   – the .pt file you saved in preprocessing
    n_splits         : int          – number of folds (default 10)
    hidden_channels  : int          – must match your model definition
    epochs           : int          – training epochs per fold
    lr               : float        – Adam learning rate
    batch_size       : int          – DataLoader batch size
    seed             : int          – controls fold split + model init
    device           : torch.device – defaults to CUDA if available

    Returns
    -------
    fold_accs : list[float]  – per-fold test accuracy
    mean_acc  : float
    std_acc   : float
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pull labels out so StratifiedKFold can balance the folds
    labels  = [data.y.item() for data in pyg_dataset]
    indices = list(range(len(pyg_dataset)))

    # Infer architecture dimensions from the saved dataset
    in_channels   = pyg_dataset[0].x.size(1)
    num_relations = int(max(data.edge_type.max().item() for data in pyg_dataset) + 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):

        print(f"\n{'─'*52}")
        print(f"  Fold {fold:2d} / {n_splits}   "
              f"(train={len(train_idx)}, test={len(test_idx)})")
        print(f"{'─'*52}")

        # Re-seed before every fold so model init is deterministic but
        # different folds still see different weight initialisations when
        # seed + fold_number vary together.
        set_seed(seed + fold)

        train_loader = DataLoader(
            [pyg_dataset[i] for i in train_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            [pyg_dataset[i] for i in test_idx],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = MutagenicityGNN(
            in_channels=in_channels,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            num_classes=2,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # Print a compact progress line every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                train_acc = evaluate(model, train_loader, device)
                print(
                    f"  Epoch {epoch:3d}/{epochs} │ "
                    f"Loss: {loss:.4f} │ Train Acc: {train_acc:.4f}"
                )

        test_acc = evaluate(model, test_loader, device)
        fold_accs.append(test_acc)
        print(f"\n  ✓ Fold {fold} Test Accuracy: {test_acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))

    print(f"\n{'═'*52}")
    print(f"  {n_splits}-Fold Cross-Validation Summary")
    print(f"{'═'*52}")
    for i, acc in enumerate(fold_accs, start=1):
        bar = "█" * int(acc * 20)
        print(f"  Fold {i:2d}: {acc:.4f}  {bar}")
    print(f"{'─'*52}")
    print(f"  Mean Accuracy : {mean_acc:.4f}")
    print(f"  Std Deviation : {std_acc:.4f}")
    print(f"  95% CI        : [{mean_acc - 2*std_acc:.4f},  {mean_acc + 2*std_acc:.4f}]")
    print(f"{'═'*52}")

    return fold_accs, mean_acc, std_acc


if __name__ == "__main__":
    set_seed(42)

    print("[INFO] Loading pre-processed PyG dataset...")
    pyg_dataset = torch.load(
        DATASET_DIR / "pyg_dataset.pt",
        weights_only=False,
    )
    print(f"[INFO] Total molecules loaded: {len(pyg_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    fold_accs, mean_acc, std_acc = run_kfold(
        pyg_dataset=pyg_dataset,
        n_splits=10,
        hidden_channels=64,   # keep identical to your train_rgcn.py
        epochs=50,            # keep identical to your train_rgcn.py
        lr=0.001,             # keep identical to your train_rgcn.py
        batch_size=32,        # keep identical to your train_rgcn.py
        seed=42,
        device=device,
    )