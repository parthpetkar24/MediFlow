import os
import sys
import torch
import joblib
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SkinCNN

# ───────────── Paths ─────────────
MODEL_PATH  = "skin_model/skin_model.pth"
LABELS_PATH = "skin_model/labels.joblib"

# ───────────── Full class names for display ─────────────
FULL_NAMES = {
    "nv" : "Melanocytic Nevi (Moles)",
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "ak" : "Actinic Keratosis",
}

# ───────────── Load ─────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = joblib.load(LABELS_PATH)
model  = SkinCNN(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ───────────── Core functions ─────────────
def predict_image(image_path):
    """Returns (label, confidence) for top prediction."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
        conf, idx = torch.max(probs, 1)
    return labels[idx.item()], conf.item()


def predict_top3(image_path):
    """Returns list of (label, confidence) for top 3."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
    top3 = torch.topk(probs, min(3, len(labels)))
    return [
        (labels[top3.indices[0][i].item()], top3.values[0][i].item())
        for i in range(len(top3.indices[0]))
    ]


# ───────────── CLI ─────────────
def main():
    print("=" * 45)
    print("     Skin Disease Predictor")
    print("=" * 45)

    image_path = input("\nEnter image path: ").strip()

    if not os.path.exists(image_path):
        print("\n❌ Image not found. Check the path and try again.")
        return

    disease, confidence = predict_image(image_path)
    full_name = FULL_NAMES.get(disease, disease)

    print("\n RESULT")
    print("─" * 40)
    print(f"  Disease    : {full_name}")
    print(f"  Confidence : {round(confidence * 100, 2)}%")

    # Confidence bar
    bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    print(f"  [{bar}]")

    if confidence < 0.50:
        print("\n    Very low confidence — result unreliable")
    elif confidence < 0.70:
        print("\n    Low confidence — consult a dermatologist")
    else:
        print("\n   Confident prediction")

    print("\n Top Predictions")
    print("─" * 40)
    for rank, (d, c) in enumerate(predict_top3(image_path), 1):
        name = FULL_NAMES.get(d, d)
        bar  = "█" * int(c * 20) + "░" * (20 - int(c * 20))
        print(f"  {rank}. {name:<35} {round(c*100, 2):>6.2f}%  [{bar}]")

    print("\n─" * 40)
    print("    AI-assisted only. Always consult a dermatologist.")
    print("─" * 40)


if __name__ == "__main__":
    main()