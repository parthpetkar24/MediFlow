import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from dataset import train_loader, test_loader, classes, num_classes
from model import SkinCNN

# ───────────── CONFIG ─────────────
EPOCHS   = 40
LR       = 1e-4
PATIENCE = 12
SAVE_DIR = "skin_model"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(SAVE_DIR, "skin_model.pth")
LABELS_PATH = os.path.join(SAVE_DIR, "labels.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {device}")
print(f"Classes ({num_classes}): {classes}\n")

# ───────────── MODEL ─────────────
model     = SkinCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ───────────── TRAINING LOOP ─────────────
best_val_acc     = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    running_loss = correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc  = 100.0 * correct / total

    # ── Validate ──
    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds  = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(test_loader)
    val_acc   = 100.0 * val_correct / val_total
    scheduler.step()

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # ── Save best / early stop ──
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# ───────────── SAVE LABELS ─────────────
joblib.dump(classes, LABELS_PATH)
print(f"\nDone! Best val acc : {best_val_acc:.2f}%")
print(f"Model  → {MODEL_PATH}")
print(f"Labels → {LABELS_PATH}")

# ───────────── PER-CLASS ACCURACY ─────────────
print("\nPer-Class Accuracy:")
print("─" * 40)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

class_correct = [0] * num_classes
class_total   = [0] * num_classes

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = torch.max(model(images), 1)
        for i in range(len(labels)):
            lbl = labels[i].item()
            class_correct[lbl] += (preds[i] == labels[i]).item()
            class_total[lbl]   += 1

for i, cls in enumerate(classes):
    if class_total[i] > 0:
        acc = 100.0 * class_correct[i] / class_total[i]
        bar = "█" * int(acc // 5) + "░" * (20 - int(acc // 5))
        print(f"  {cls:<6} {bar} {acc:.1f}%  ({class_correct[i]}/{class_total[i]})")