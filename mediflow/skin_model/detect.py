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

# ───────────── Full class names ─────────────
FULL_NAMES = {
    "nv" : "Melanocytic Nevi (Moles)",
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "ak" : "Actinic Keratosis",
}

# ───────────── Load model ─────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = joblib.load(LABELS_PATH)
model  = SkinCNN(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ───────────── Transform ─────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ───────────── TTA Prediction ─────────────
def predict(image_path):
    img = Image.open(image_path).convert("RGB")

    tta_transforms = [
        transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.Resize((224, 224)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.Resize((240, 240)),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.Resize((224, 224)),
                             transforms.RandomRotation(degrees=(10, 10)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([transforms.Resize((224, 224)),
                             transforms.RandomRotation(degrees=(-10, -10)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    ]

    probs_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            inp   = t(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(inp), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = probs_sum / len(tta_transforms)

    # Top 3
    top3  = torch.topk(probs_avg, min(3, len(labels)))
    top3_results = [
        (labels[top3.indices[0][i].item()], top3.values[0][i].item())
        for i in range(len(top3.indices[0]))
    ]

    return top3_results


# ───────────── Confidence label ─────────────
def confidence_label(conf):
    if conf >= 0.80:
        return "High Confidence "
    elif conf >= 0.60:
        return "Moderate Confidence "
    else:
        return "Low Confidence "


# ───────────── Main ─────────────
def main():
    print("=" * 50)
    print("        Skin Disease Predictor")
    print("=" * 50)

    while True:
        image_path = input("\nEnter image path (or 'q' to quit): ").strip()

        if image_path.lower() == 'q':
            print("\nGoodbye!")
            break

        if not os.path.exists(image_path):
            print(" Image not found. Check the path and try again.")
            continue

        try:
            results = predict(image_path)

            top_disease, top_conf = results[0]
            full_name = FULL_NAMES.get(top_disease, top_disease)

            print("\n" + "=" * 50)
            print("  RESULT")
            print("=" * 50)
            print(f"  Disease    : {full_name}")
            print(f"  Confidence : {round(top_conf * 100, 2)}%")
            bar = "█" * int(top_conf * 20) + "░" * (20 - int(top_conf * 20))
            print(f"  Accuracy   : [{bar}] {confidence_label(top_conf)}")

            print("\n" + "─" * 50)
            print("  TOP 3 PREDICTIONS")
            print("─" * 50)
            for rank, (disease, conf) in enumerate(results, 1):
                name = FULL_NAMES.get(disease, disease)
                bar  = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                print(f"  {rank}. {name:<35} {round(conf * 100, 2):>6.2f}%")
                print(f"     [{bar}]")

            print("\n" + "─" * 50)
            if top_conf < 0.60:
                print("  Low confidence — image may be unclear")
                print("  Please consult a dermatologist")
            else:
                print("AI assisted prediction only.")
                print("Always consult a dermatologist.")
            print("─" * 50)

        except Exception as e:
            print(f" Error processing image: {e}")


if __name__ == "__main__":
    main()