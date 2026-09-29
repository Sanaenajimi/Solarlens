"""
Export an already-trained SolarLens checkpoint to ONNX — VERSION WEB.
Run this AFTER training — it does NOT retrain anything, it just
loads the saved .pth checkpoint and converts it to ONNX for use
in the browser with ONNX Runtime Web.

IMPORTANT: this uses the legacy exporter (dynamo=False) to force a
SINGLE .onnx file. The new dynamo-based exporter can split large
models into model.onnx + model.onnx.data, which is awkward to serve
correctly to a browser (two files, exact same folder required,
extra fetch). ResNet-50 (~100MB) fits comfortably under the legacy
exporter's 2GB single-file limit, so we force it explicitly.

We also bake softmax into the exported graph so the browser receives
probabilities directly (0 to 1, summing to 1) instead of raw logits —
one less thing to get wrong in JavaScript.

Usage:
    python export_onnx.py --checkpoint checkpoints/solarlens_best.pth
"""

import argparse
import json
import torch
import torch.nn as nn
from models.solarlens_model import SolarLensClassifier


class SolarLensWebWrapper(nn.Module):
    """Wraps the classifier to output softmax probabilities directly,
    so app.js can use the output with no extra math."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=1)


def main(args):
    device = torch.device('cpu')

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A')}")

    model = SolarLensClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    web_model = SolarLensWebWrapper(model)
    web_model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    print(f"Exporting to {args.output} (legacy exporter, single file)...")
    torch.onnx.export(
        web_model, dummy, args.output,
        input_names=['image'],
        output_names=['probabilities'],
        dynamic_axes={'image': {0: 'batch'}, 'probabilities': {0: 'batch'}},
        opset_version=17,
        dynamo=False,   # <-- forces the legacy exporter, single .onnx file
    )
    print(f"✓ Done. Model exported to {args.output}")

    # Save the class names alongside the model — app.js needs this
    # ordering to know which probability index means which defect.
    labels_path = args.output.rsplit('.', 1)[0] + '_labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"✓ Class labels saved to {labels_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export SolarLens checkpoint to ONNX (web-ready)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/solarlens_best.pth')
    parser.add_argument('--output', type=str, default='checkpoints/solarlens_web.onnx')
    args = parser.parse_args()
    main(args)

