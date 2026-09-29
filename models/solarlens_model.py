"""
╔══════════════════════════════════════════════════════════════╗
║  SolarLens — ResNet-50 + SE Attention Block                 ║
║  6-class Solar Panel Defect Classification                  ║
║  Author: Sanae Najimi                                       ║
╚══════════════════════════════════════════════════════════════╝

Classes:
  0 - Clean
  1 - Dusty
  2 - Snow-Covered
  3 - Bird Dropping
  4 - Electrical Fault (Hotspot, Diode Bypass, PID)
  5 - Physical Damage (Micro-Crack, Snail Trail, Delamination)

Datasets:
  - PV Panel Defect Dataset (Kaggle): https://kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
  - InfraredSolarModules (20K IR images): https://github.com/InfraredSolarModules
  - PVEL-AD (36K EL images): https://github.com/binyisu/PVEL-AD
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    Learns to recalibrate channel responses adaptively.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)


class SolarLensClassifier(nn.Module):
    """
    ResNet-50 backbone with SE attention for solar panel defect classification.
    
    Architecture:
        Input (224×224×3)
        → ResNet-50 backbone (layers 1-3 frozen, layer 4 fine-tuned)
        → SE Attention Block (channel recalibration)
        → Global Average Pooling
        → Dropout (p=0.4)
        → FC(2048, 512) → ReLU → BatchNorm
        → Dropout (p=0.3)
        → FC(512, num_classes)
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        # Load pretrained ResNet-50
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Freeze layers 1-3 for transfer learning
        if freeze_backbone:
            for param in list(self.conv1.parameters()) + list(self.bn1.parameters()):
                param.requires_grad = False
            for layer in [self.layer1, self.layer2, self.layer3]:
                for param in layer.parameters():
                    param.requires_grad = False

        # SE Attention after layer4
        self.se_block = SEBlock(channels=2048, reduction=16)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # SE attention
        x = self.se_block(x)

        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Extract intermediate feature maps for Grad-CAM visualization."""
        features = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        x = self.layer4(x)
        features['layer4'] = x

        x = self.se_block(x)
        features['se_output'] = x

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        features['logits'] = logits

        return features


# ──────────────────────────────────────────────────────────────
# Grad-CAM Implementation
# ──────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Visualizes which regions of the input image the model focuses on.
    """

    def __init__(self, model: SolarLensClassifier, target_layer: str = 'layer4'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target = getattr(model, target_layer)
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weighted activation map
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze()


# ──────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────
def get_transforms(mode: str = 'train'):
    """Data augmentation and normalization transforms."""
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }


if __name__ == '__main__':
    # Quick test
    model = SolarLensClassifier(num_classes=6, pretrained=False)
    params = count_parameters(model)
    print(f"SolarLens Model Summary:")
    print(f"  Total parameters:     {params['total_M']:.1f}M")
    print(f"  Trainable parameters: {params['trainable_M']:.1f}M")
    print(f"  Frozen parameters:    {params['total_M'] - params['trainable_M']:.1f}M")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Predictions:  {torch.softmax(out, dim=1).detach().numpy().round(3)}")
