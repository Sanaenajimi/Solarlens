"""
╔══════════════════════════════════════════════════════════════╗
║  SolarLens — Training Script                                ║
║  Fine-tune ResNet-50 + SE on PV Panel Defect Dataset        ║
║  Author: Sanae Najimi                                       ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python train.py --data_dir ./data/pv-panel-defect --epochs 100 --batch_size 16 --patience 15

Datasets (download manually from Kaggle):
    1. PV Panel Defect Dataset:
       https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
       → Extract to ./data/pv-panel-defect/

    2. Solar Panel Images (Clean + Faulty):
       https://www.kaggle.com/datasets/pythonafroz/solar-panel-images
       → Extract to ./data/solar-panel-images/

    3. InfraredSolarModules (optional, IR):
       https://www.kaggle.com/datasets/marcosgabriel/photovoltaic-system-thermography
       → Extract to ./data/infrared-solar/

Directory structure expected:
    data/pv-panel-defect/
    ├── Clean/
    ├── Dusty/
    ├── Snow-Covered/
    ├── Bird-Dropping/
    ├── Electrical-Damage/
    └── Physical-Damage/
"""

import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from models.solarlens_model import SolarLensClassifier, get_transforms, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx+1}/{len(loader)}] | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%")

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100. * correct / total, all_preds, all_labels


def main(args):
    print("=" * 60)
    print("  SolarLens — Training Pipeline")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data — auto-detect dataset structure
    # ─────────────────────────────────────────────────────────
    # Handles TWO common Kaggle layouts:
    #
    #   Layout A (pre-split):          Layout B (flat classes):
    #   data/                          data/
    #   ├── train/                     ├── Clean/
    #   │   ├── Clean/                 ├── Dusty/
    #   │   ├── Dusty/                 ├── Snow-Covered/
    #   │   └── ...                    └── ...
    #   ├── val/
    #   └── test/
    # ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.data_dir}")
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')

    # CPU-friendly DataLoader settings
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        'num_workers': 4 if use_cuda else 0,   # Windows CPU: 0 avoids multiprocessing bugs
        'pin_memory': use_cuda,                  # pin_memory only useful on GPU
    }

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    if os.path.isdir(train_dir):
        # ── Layout A: pre-split folders ──
        print("  → Detected pre-split structure (train/val/test folders)")
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        class_names = train_set.classes
        num_classes = len(class_names)

        if os.path.isdir(val_dir):
            val_set = datasets.ImageFolder(val_dir, transform=val_transform)
        else:
            # No val folder → split 90/10 from train
            n_val = int(0.1 * len(train_set))
            train_set, val_set = random_split(train_set, [len(train_set) - n_val, n_val],
                                               generator=torch.Generator().manual_seed(42))
            val_set.dataset.transform = val_transform

        if os.path.isdir(test_dir):
            test_set = datasets.ImageFolder(test_dir, transform=val_transform)
        else:
            test_set = val_set  # fallback: test = val

        n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)

    else:
        # ── Layout B: flat class folders → auto-split ──
        print("  → Detected flat structure (class folders at root)")
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
        class_names = full_dataset.classes
        num_classes = len(class_names)

        n_total = len(full_dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val

        train_set, val_set, test_set = random_split(
            full_dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        val_set.dataset.transform = val_transform
        test_set.dataset.transform = val_transform

    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Model
    model = SolarLensClassifier(num_classes=num_classes, pretrained=True, freeze_backbone=True).to(device)
    params = count_parameters(model)
    print(f"\nModel: ResNet-50 + SE Attention")
    print(f"  Total params:     {params['total_M']:.1f}M")
    print(f"  Trainable params: {params['trainable_M']:.1f}M")

    # Class weights for imbalanced data
    # Computed from TRAIN split only (not val/test) — train_set can be
    # either a plain ImageFolder (Layout A) or a Subset wrapping one
    # (Layout B, after random_split), so we handle both cases.
    def get_labels(dataset):
        if hasattr(dataset, 'samples'):          # plain ImageFolder
            return [label for _, label in dataset.samples]
        elif hasattr(dataset, 'dataset'):        # torch.utils.data.Subset
            base_samples = dataset.dataset.samples
            return [base_samples[i][1] for i in dataset.indices]
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    train_labels = get_labels(train_set)
    class_counts = [0] * num_classes
    for label in train_labels:
        class_counts[label] += 1

    print(f"  Class distribution (train): "
          f"{dict(zip(class_names, class_counts))}")

    weights = torch.FloatTensor([
        len(train_labels) / (num_classes * max(c, 1)) for c in class_counts
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer: different LR for backbone vs head
    backbone_params = list(model.layer4.parameters()) + list(model.se_block.parameters())
    head_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")

        # ── Progressive Unfreezing ──────────────────────────
        # Phase 1 (epochs 1-N): only layer4 + SE + classifier
        # Phase 2 (epochs N+): unfreeze layer3 with lower LR
        # Rationale: on a small dataset, early layers hold generic
        # features (edges, textures) that transfer perfectly from
        # ImageNet. Layer3/4 encode higher-level patterns that need
        # domain adaptation to PV-specific defect morphology.
        if epoch == args.unfreeze_epoch:
            print(f"\n  ★ PROGRESSIVE UNFREEZING: layer3 now trainable")
            for param in model.layer3.parameters():
                param.requires_grad = True
            # Add layer3 params to optimizer with very low LR
            optimizer.add_param_group({
                'params': model.layer3.parameters(),
                'lr': args.lr * 0.01,  # 100x lower than head
            })
            new_params = count_parameters(model)
            print(f"    Trainable params: {new_params['trainable_M']:.1f}M (was {params['trainable_M']:.1f}M)\n")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lr = optimizer.param_groups[1]['lr']
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%")
        print(f"  LR: {lr:.6f}")

        # ── Early Stopping on val_loss ──────────────────────
        # We track val_loss (not val_acc) because loss is smoother
        # and catches overfitting before accuracy drops.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  ⚠ No val_loss improvement for {epochs_no_improve}/{args.patience} epochs")

        if epochs_no_improve >= args.patience:
            print(f"\n  ✋ EARLY STOPPING at epoch {epoch} (patience={args.patience})")
            print(f"     Best val_loss: {best_val_loss:.4f} | Best val_acc: {best_val_acc:.1f}%")
            break

        # Save best model (by accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
            }, 'checkpoints/solarlens_best.pth')
            print(f"  ★ New best model saved (val_acc={val_acc:.1f}%)")

        print()

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"  Final Test Evaluation")
    print(f"{'='*60}\n")

    checkpoint = torch.load('checkpoints/solarlens_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")

    # Classification report
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print(f"\nClassification Report:")
        print(classification_report(labels, preds, target_names=class_names))
        print(f"Confusion Matrix:")
        print(confusion_matrix(labels, preds))
    except ImportError:
        print("(Install scikit-learn for detailed report)")

    # Save history
    with open('checkpoints/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Export to ONNX for production deployment
    print(f"\nExporting to ONNX...")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy, 'checkpoints/solarlens.onnx',
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17,
    )
    print("Model exported to checkpoints/solarlens.onnx")
    print("\n✓ Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SolarLens Training')
    parser.add_argument('--data_dir', type=str, default='./data/pv-panel-defect')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs (early stopping will likely trigger before)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='16 recommended for small datasets (<2K images); '
                             'use 8 if GPU memory allows for even better regularization')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (epochs without val improvement)')
    parser.add_argument('--unfreeze_epoch', type=int, default=20,
                        help='Epoch at which to unfreeze layer3 for deeper fine-tuning')
    args = parser.parse_args()
    main(args)
