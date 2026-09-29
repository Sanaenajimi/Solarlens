"""
Quick sanity check: verify the exported ONNX model loads and runs.
Run this after export_onnx.py to confirm the .onnx file is valid.

Usage:
    python verify_onnx.py
"""

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime first: pip install onnxruntime")
    exit(1)

session = ort.InferenceSession("checkpoints/solarlens.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input : {input_name} {session.get_inputs()[0].shape}")
print(f"Output: {output_name} {session.get_outputs()[0].shape}")

# Dummy image batch (same shape as training: 1x3x224x224)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: dummy_input})

logits = result[0]
probs = np.exp(logits) / np.exp(logits).sum()
print(f"\nLogits: {logits.round(2)}")
print(f"Probs:  {probs.round(3)}")
print("\n✓ ONNX model runs correctly.")
