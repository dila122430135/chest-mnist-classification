import torch
import torch.nn as nn
from torchvision import models

class DenseNet121(nn.Module):
    """
    Wrapper DenseNet121 yang kompatibel dengan proyek.
    - Jika input grayscale (in_channels==1) maka channel di-repeat ke 3.
    - Jika num_classes == 2, output dim = 1 (untuk BCEWithLogitsLoss).
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # coba gunakan API weights (torchvision baru) atau fallback ke pretrained arg lama
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
        except Exception:
            backbone = models.densenet121(pretrained=pretrained)

        in_feat = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_feat, 1 if num_classes == 2 else num_classes)
        self.backbone = backbone

    def forward(self, x):
        # pastikan input punya 3 channel jika backbone pretrained
        if self.in_channels == 1 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

# --- Pengujian singkat ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    print("--- Menguji Model 'DenseNet121' ---")
    model = DenseNet121(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False)
    print(model)

    dummy_input = torch.randn(4, IN_CHANNELS, 224, 224)  # DenseNet biasanya pakai 224x224
    output = model(dummy_input)
    print(f"Ukuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")