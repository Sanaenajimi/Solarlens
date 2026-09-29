"""
Loads the trained SolarLens model for inference.

Priority order:
  1. Local checkpoint (checkpoints/solarlens_inference.pth or
     checkpoints/solarlens_best.pth) — used when running locally.
     NO Hugging Face, NO internet needed at all.
  2. Hugging Face Hub — only used as a fallback, for later when the
     app is deployed online (e.g. Streamlit Cloud) where the .pth
     isn't committed to the GitHub repo (100MB size limit).

For now, running locally: this works with ZERO Hugging Face setup.
"""

import os
import streamlit as st
import torch

from models.solarlens_model import SolarLensClassifier

# Tries these local paths in order — whichever exists first is used
LOCAL_CANDIDATES = [
    "checkpoints/solarlens_inference.pth",
    "checkpoints/solarlens_best.pth",
]

HF_REPO_ID = "Snjm/solarlens-resnet50"
HF_FILENAME = "solarlens_inference.pth"


@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """
    Returns (model, class_names).
    st.cache_resource means this only runs ONCE per app session,
    not on every Streamlit interaction/rerun.
    """
    checkpoint_path = None
    for path in LOCAL_CANDIDATES:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        # Fallback: download from Hugging Face (only needed once deployed online)
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]

    model = SolarLensClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


if __name__ == "__main__":
    # Standalone test — run: python utils/model_loader.py
    found = False
    for path in LOCAL_CANDIDATES:
        if os.path.exists(path):
            print(f"✓ Found local checkpoint: {path}")
            checkpoint = torch.load(path, map_location="cpu")
            print(f"  Classes: {checkpoint['class_names']}")
            print(f"  Val acc: {checkpoint.get('val_acc')}")
            found = True
            break
    if not found:
        print(f"✗ No local checkpoint found in {LOCAL_CANDIDATES}")
        print(f"  Would fall back to Hugging Face: {HF_REPO_ID}")
