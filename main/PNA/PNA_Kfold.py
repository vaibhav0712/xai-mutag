from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import numpy as np
import torch

from PNA.PNA_Train import (
    pyg_dataset,
    get_fresh_model,
    train_epoch,
    evaluate,
    device,
    set_seed,
)

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001

set_seed(0)

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=0,
)

fold_accs = []

for fold, (train_idx, test_idx) in enumerate(kf.split(pyg_dataset), start=1):

    print("=" * 60)
    print(f"Fold {fold}/10")
    print("=" * 60)

    train_loader = DataLoader(
        [pyg_dataset[i] for i in train_idx],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        [pyg_dataset[i] for i in test_idx],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = get_fresh_model("pna")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=5e-4,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):

        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        if epoch % 5 == 0 or epoch == 1:
            train_acc = evaluate(model, train_loader, device)
            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {loss:.4f} | "
                f"Train Acc: {train_acc:.4f}"
            )

    test_acc = evaluate(model, test_loader, device)

    print(f"Fold {fold} Test Accuracy: {test_acc:.4f}")

    fold_accs.append(test_acc)

print("\n" + "=" * 60)
print("10-FOLD RESULTS")
print("=" * 60)

for i, acc in enumerate(fold_accs, start=1):
    print(f"Fold {i}: {acc:.4f}")

print(f"\nMean Accuracy : {np.mean(fold_accs):.4f}")
print(f"Std Accuracy  : {np.std(fold_accs):.4f}")