import torch
import torch.nn as nn
import os
import argparse
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch_geometric.seed import seed_everything

# Import required components from the Train file
from RGCN_RDF_Train import (
    RGCN,
    data,
    device,
    relation_to_id,
    train_idx,
    train_y,
    test_idx,
    test_y
)

def main(seed=42):
    # Set seed and configure deterministic settings
    seed_everything(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Setting up 10-Fold Cross-Validation with seed {seed}...")
    
    # Combine the existing train and test indices & labels to form the complete dataset
    all_idx = torch.cat([train_idx, test_idx], dim=0)
    all_y = torch.cat([train_y, test_y], dim=0)
    
    # Convert tensors to numpy for compatibility with sklearn splitter
    all_idx_np = all_idx.cpu().numpy()
    all_y_np = all_y.cpu().numpy()
    
    # 10-Fold Stratified Split
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_accuracies = []
    
    # Make sure graph data is on the correct device
    graph_data = data.to(device)
    in_channels = graph_data.x.shape[1]
    num_rels = len(relation_to_id)
    
    print(f"Total labeled instances: {len(all_idx)}")
    
    for fold, (train_split_idx, val_split_idx) in enumerate(skf.split(all_idx_np, all_y_np)):
        print(f"\n--- Fold {fold + 1} / 10 ---")
        
        # Get PyTorch tensors for the current fold splits
        fold_train_idx = torch.tensor(all_idx_np[train_split_idx], dtype=torch.long, device=device)
        fold_train_y = torch.tensor(all_y_np[train_split_idx], dtype=torch.long, device=device)
        fold_val_idx = torch.tensor(all_idx_np[val_split_idx], dtype=torch.long, device=device)
        fold_val_y = torch.tensor(all_y_np[val_split_idx], dtype=torch.long, device=device)
        
        # Initialize a fresh model instance for this fold
        model = RGCN(
            in_channels=in_channels,
            num_rels=num_rels,
            num_classes=2,
            hidden_dim=16
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Train for 70 epochs (matching RGCN_RDF_Train.py)
        for epoch in range(1, 71):
            model.train()
            optimizer.zero_grad()
            out = model(graph_data.x, graph_data.edge_index, graph_data.edge_type)
            loss = criterion(out[fold_train_idx], fold_train_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0 or epoch == 70:
                # Calculate training accuracy and validation accuracy
                model.eval()
                with torch.no_grad():
                    train_out = model(graph_data.x, graph_data.edge_index, graph_data.edge_type)
                    train_pred = train_out[fold_train_idx].argmax(dim=-1)
                    train_acc = (train_pred == fold_train_y).sum().item() / len(fold_train_y)
                    
                    val_pred = train_out[fold_val_idx].argmax(dim=-1)
                    val_acc = (val_pred == fold_val_y).sum().item() / len(fold_val_y)
                
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Final validation accuracy for the current fold
        fold_accuracies.append(val_acc)
        print(f"Fold {fold + 1} finished. Final Val Acc: {val_acc:.4f}")
        
    print("\n==========================================")
    print("K-Fold Cross-Validation Results Summary:")
    for f_idx, acc in enumerate(fold_accuracies):
        print(f"Fold {f_idx + 1}: {acc:.4f}")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Fold Cross Validation on RGCN model.")
    parser.add_argument('--seed', type=int, default=42, help="Seed value for reproducibility.")
    args = parser.parse_args()
    main(seed=args.seed)
