"""
Utility to run the trained SolarLens model on a folder of real images.
This is the honest inference module — it does NOT simulate anything,
it just walks the filesystem and calls the trained model.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

from utils.model_loader import load_model
from models.solarlens_model import get_transforms


def list_test_images(base_dir: str = "data/pv-panel-defect/test") -> dict:
    """
    Returns {class_name: [list of image paths]} for the test folder.
    If the folder doesn't exist, returns an empty dict.
    """
    if not os.path.isdir(base_dir):
        return {}

    result = {}
    for class_name in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = sorted(
            glob.glob(os.path.join(class_dir, "*.jpg"))
            + glob.glob(os.path.join(class_dir, "*.jpeg"))
            + glob.glob(os.path.join(class_dir, "*.png"))
        )
        if images:
            result[class_name] = images
    return result


@st.cache_data(show_spinner=False)
def predict_image(image_path: str) -> dict:
    """
    Runs the trained model on a single image and returns the full prediction.
    st.cache_data means each image is only inferenced ONCE per session.
    """
    model, class_names = load_model()
    transform = get_transforms('val')

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        'path': image_path,
        'class_names': class_names,
        'probs': probs.tolist(),
        'pred_idx': pred_idx,
        'pred_class': class_names[pred_idx],
        'confidence': float(probs[pred_idx]) * 100,
    }


@st.cache_data(show_spinner="Running model on all test images...")
def predict_all_test_images(base_dir: str = "data/pv-panel-defect/test") -> list:
    """
    Runs the model on the ENTIRE test folder and returns a list of results
    with true label + predicted label for each image.
    Cached so we only pay the inference cost once per session.
    """
    images_by_class = list_test_images(base_dir)
    if not images_by_class:
        return []

    results = []
    for true_class, paths in images_by_class.items():
        for path in paths:
            pred = predict_image(path)
            pred['true_class'] = true_class
            pred['correct'] = (pred['pred_class'] == true_class)
            results.append(pred)
    return results
