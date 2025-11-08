import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
from model_resnet import ResNet18
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter (sesuai permintaan) ---
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 3
IMAGE_SIZE = (224, 224)

# Pilih model: 'simple' atau 'resnet'
MODEL_CHOICE = 'resnet'  # ubah ke 'simple' untuk SimpleCNN

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)

    # 2. Inisialisasi Model (gunakan pretrained torchvision ResNet jika PRETRAINED True)
    use_torchvision_resnet = False
    if MODEL_CHOICE == 'resnet':
        if PRETRAINED:
            try:
                from torchvision import models
                # kompatibilitas API weights/name across torchvision versions
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                    backbone = models.resnet18(weights=weights)
                except Exception:
                    backbone = models.resnet18(pretrained=True)
                # ganti classifier sesuai jumlah kelas
                in_feat = backbone.fc.in_features
                backbone.fc = nn.Linear(in_feat, 1 if num_classes == 2 else num_classes)
                model = backbone
                use_torchvision_resnet = True
            except Exception:
                # fallback ke ResNet18 lokal jika torchvision tidak tersedia
                model = ResNet18(in_channels=in_channels, num_classes=num_classes)
                use_torchvision_resnet = False
        else:
            model = ResNet18(in_channels=in_channels, num_classes=num_classes)
            use_torchvision_resnet = False
    else:
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Freeze backbone untuk beberapa epoch (train head dulu) jika menggunakan pretrained torchvision ResNet
    if PRETRAINED and FREEZE_BACKBONE_EPOCHS > 0:
        for name, param in model.named_parameters():
            # tetap latih layer akhir (nama berbeda tergantung model)
            if ('fc' in name) or ('classifier' in name) or ('head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 3. Loss + optimizer + scheduler
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # history
    train_losses_history, val_losses_history = [], []
    train_accs_history, val_accs_history = [], []

    print("\n--- Memulai Training ---")

    # helper untuk memastikan shape (N,1) untuk label biner dan output
    def ensure_binary_shape(t):
        t = t.to(device).float()
        if t.dim() == 3 and t.size(1) == 1 and t.size(2) == 1:
            t = t.view(t.size(0), 1)
        elif t.dim() == 2 and t.size(1) != 1:
            t = t.view(t.size(0), 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        return t

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # unfreeze logic at start of epoch if reached freeze period
        if PRETRAINED and (epoch + 1) == FREEZE_BACKBONE_EPOCHS:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            print(f"Epoch {epoch+1}: Unfrozen backbone, rebuilt optimizer.")

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # jika menggunakan torchvision pretrained ResNet, resize and convert grayscale->3ch
            if use_torchvision_resnet:
                if images.dim() == 3 or images.size(2) != IMAGE_SIZE[0] or images.size(3) != IMAGE_SIZE[1]:
                    images = F.interpolate(images, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
                if in_channels == 1 and images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)

            outputs = model(images)

            if num_classes == 2:
                outputs = outputs.view(outputs.size(0), -1)  # (N,1)
                labels_t = ensure_binary_shape(labels)        # (N,1)
            else:
                labels_t = labels.to(device).long()

            loss = criterion(outputs, labels_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if num_classes == 2:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (preds == labels_t).sum().item()
                train_total += labels_t.size(0)
            else:
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels_t).sum().item()
                train_total += labels.size(0)

        avg_train_loss = running_loss / max(len(train_loader), 1)
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0.0

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                if use_torchvision_resnet:
                    if images.dim() == 3 or images.size(2) != IMAGE_SIZE[0] or images.size(3) != IMAGE_SIZE[1]:
                        images = F.interpolate(images, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
                    if in_channels == 1 and images.size(1) == 1:
                        images = images.repeat(1, 3, 1, 1)

                outputs = model(images)
                if num_classes == 2:
                    outputs = outputs.view(outputs.size(0), -1)
                    labels_t = ensure_binary_shape(labels)
                else:
                    labels_t = labels.to(device).long()

                val_loss = criterion(outputs, labels_t)
                val_running_loss += val_loss.item()

                if num_classes == 2:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (preds == labels_t).sum().item()
                    val_total += labels_t.size(0)
                else:
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels_t).sum().item()
                    val_total += labels.size(0)

        avg_val_loss = val_running_loss / max(len(val_loader), 1)
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0

        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)

        # scheduler step menggunakan validation loss
        scheduler_plateau.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")

    # plot & visualize
    plot_training_history(train_losses_history, val_losses_history, train_accs_history, val_accs_history)
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)


if __name__ == '__main__':
    train()
