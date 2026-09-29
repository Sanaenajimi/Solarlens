"""
Strip the optimizer state from a training checkpoint to create a
lightweight, GitHub-friendly inference-only checkpoint.

The full training checkpoint (.pth) contains model weights + optimizer
state (Adam keeps 2x extra params for momentum/variance), which roughly
triples the file size. For deployment/portfolio we only need the model
weights (~100MB for ResNet-50), not the optimizer state.

Usage:
    python strip_checkpoint.py
"""

import torch
import os

INPUT_PATH = "checkpoints/solarlens_best.pth"
OUTPUT_PATH = "checkpoints/solarlens_inference.pth"

print(f"Loading: {INPUT_PATH}")
checkpoint = torch.load(INPUT_PATH, map_location="cpu")

lightweight = {
    "model_state_dict": checkpoint["model_state_dict"],
    "class_names": checkpoint["class_names"],
    "val_acc": checkpoint.get("val_acc"),
}

torch.save(lightweight, OUTPUT_PATH)

before_mb = os.path.getsize(INPUT_PATH) / 1e6
after_mb = os.path.getsize(OUTPUT_PATH) / 1e6

print(f"Before: {before_mb:.1f} MB")
print(f"After:  {after_mb:.1f} MB")
print(f"✓ Saved lightweight checkpoint to {OUTPUT_PATH}")

if after_mb > 100:
    print("\n⚠ Still over GitHub's 100MB limit.")
    print("  → Use Git LFS, or host the model on Hugging Face Hub")
    print("    and download it at app startup instead of committing it.")
else:
    print("\n✓ Under GitHub's 100MB limit — safe to commit.")
