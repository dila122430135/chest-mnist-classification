import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model_densenet import DenseNet121
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameters ---
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 2
IMAGE_SIZE = (224, 224)

def train():
    # load data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)

    # model
    model = DenseNet121(in_channels=in_channels, num_classes=num_classes, pretrained=PRETRAINED)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # optionally freeze backbone for a few epochs (train classifier only)
    if PRETRAINED and FREEZE_BACKBONE_EPOCHS > 0:
        for name, param in model.named_parameters():
            # keep classifier (DenseNet uses backbone.classifier) trainable
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # loss / optimizer / scheduler
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def to_binary_tensor(t):
        t = t.to(device).float()
        if t.dim() == 3 and t.size(1) == 1 and t.size(2) == 1:
            t = t.view(t.size(0), 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        return t

    print("\n--- Memulai Training DenseNet121 ---")
    for epoch in range(EPOCHS):
        # unfreeze after freeze epochs
        if PRETRAINED and (epoch == FREEZE_BACKBONE_EPOCHS):
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            print(f"Epoch {epoch+1}: backbone unfrozen, optimizer rebuilt.")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # resize and convert grayscale->3ch if needed
            if imgs.dim() == 3 or imgs.size(2) != IMAGE_SIZE[0] or imgs.size(3) != IMAGE_SIZE[1]:
                imgs = F.interpolate(imgs, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
            if in_channels == 1 and imgs.size(1) == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            outputs = model(imgs)
            if num_classes == 2:
                outputs = outputs.view(outputs.size(0), -1)     # (N,1)
                labels_t = to_binary_tensor(labels)             # (N,1)
            else:
                labels_t = labels.long().to(device)

            loss = criterion(outputs, labels_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            if num_classes == 2:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels_t).sum().item()
                total += labels_t.size(0)
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_t).sum().item()
                total += labels.size(0)

        avg_train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        # validation
        model.eval()
        val_running = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                if imgs.dim() == 3 or imgs.size(2) != IMAGE_SIZE[0] or imgs.size(3) != IMAGE_SIZE[1]:
                    imgs = F.interpolate(imgs, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
                if in_channels == 1 and imgs.size(1) == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)

                outputs = model(imgs)
                if num_classes == 2:
                    outputs = outputs.view(outputs.size(0), -1)
                    labels_t = to_binary_tensor(labels)
                else:
                    labels_t = labels.long().to(device)

                vloss = criterion(outputs, labels_t)
                val_running += vloss.item() * imgs.size(0)

                if num_classes == 2:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (preds == labels_t).sum().item()
                    val_total += labels_t.size(0)
                else:
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels_t).sum().item()
                    val_total += labels.size(0)

        avg_val_loss = val_running / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print("--- Training Selesai ---")

    # plot & visualize
    plot_training_history(history['train_loss'], history['val_loss'],
                         history['train_acc'], history['val_acc'])
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()