from django.shortcuts import render
import torch
import joblib
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import sys

# Disease Prediction
DISEASE_MODEL_PATH = "disease_predictor/disease_model.pkl"
SYMPTOM_ENCODER_PATH = "disease_predictor/symptom_encoder.pkl"
DISEASE_ENCODER_PATH = "disease_predictor/disease_encoder.pkl"

disease_model = joblib.load(DISEASE_MODEL_PATH)
symptom_encoder = joblib.load(SYMPTOM_ENCODER_PATH)
disease_encoder = joblib.load(DISEASE_ENCODER_PATH)

VALID_SYMPTOMS = set(symptom_encoder.classes_)

# Skin Disease Prediction
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_model'))
from skin_model.model import SkinCNN

MODEL_PATH  = 'skin_model/skin_model.pth'
LABELS_PATH = 'skin_model/labels.joblib'

FULL_NAMES = {
    "nv" : "Melanocytic Nevi (Moles)",
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "ak" : "Actinic Keratosis",
}

RISK_LEVEL = {
    "nv" : ("Low",    "green"),
    "mel": ("High",   "red"),
    "bcc": ("High",   "red"),
    "bkl": ("Low",    "green"),
    "ak" : ("Medium", "yellow"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = joblib.load(LABELS_PATH)
skin_model = SkinCNN(len(labels))
skin_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
skin_model.to(device)
skin_model.eval()


def predict_skin(image):
    tta_transforms = [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((240, 240)), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(10,10)),
                             transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation(degrees=(-10,-10)),
                             transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]

    probs_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            inp   = t(image).unsqueeze(0).to(device)
            probs = torch.softmax(skin_model(inp), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = probs_sum / len(tta_transforms)
    top3 = torch.topk(probs_avg, min(3, len(labels)))

    results = []
    for i in range(len(top3.indices[0])):
        lbl  = labels[top3.indices[0][i].item()]
        conf = round(top3.values[0][i].item() * 100, 2)
        results.append({
            "label"     : lbl,
            "full_name" : FULL_NAMES.get(lbl, lbl),
            "confidence": conf,
            "risk"      : RISK_LEVEL.get(lbl, ("Unknown", "gray"))[0],
            "risk_color": RISK_LEVEL.get(lbl, ("Unknown", "gray"))[1],
        })
    return results


def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def disease_predictor(request):
    context={}
    # send symptoms to frontend
    context["all_symptoms"]=sorted(VALID_SYMPTOMS)
    if request.method=="POST":
        symptoms=request.POST.getlist("symptoms")
        # remove invalid symptoms
        symptoms=[
            s for s in symptoms
            if s in VALID_SYMPTOMS
        ]
        context["selected"]=symptoms
        if len(symptoms)<2:
            context["need_more"]="Give more details"
            return render(
                request,
                "disease-predictor.html",
                context
            )
        try:
            encoded=symptom_encoder.transform([symptoms])
            prediction=disease_model.predict(encoded)
            probabilities=disease_model.predict_proba(encoded)
            disease=disease_encoder.inverse_transform(prediction)[0]
            confidence=round(
                np.max(probabilities)*100,2
            )
            # Top 3 predictions
            top3=np.argsort(
                probabilities[0]
            )[-3:][::-1]
            top_diseases=[]
            for i in top3:
                top_diseases.append({
                    "name":
                    disease_encoder.inverse_transform([i])[0],
                    "confidence":
                    round(
                        probabilities[0][i]*100,2
                    )
                })
            context["predicted_disease"]=disease
            context["confidence"]=confidence
            context["top_diseases"]=top_diseases
        except:
            context["need_more"]="Give more details"
    return render(
        request,
        "disease-predictor.html",
        context
    )

def skin_disease_predictor(request):
    context = {}

    if request.method == "POST" and request.FILES.get("skin_image"):
        try:
            img_file = request.FILES["skin_image"]
            image    = Image.open(img_file).convert("RGB")
            results  = predict_skin(image)
            top      = results[0]

            context["results"]   = results
            context["top"]       = top
            context["bar_width"] = int(top["confidence"])

            #  Confidence — all classes decided in Python
            if top["confidence"] >= 80:
                context["confidence_label"]      = "High Confidence"
                context["confidence_bar_class"]  = "bg-green-500"
                context["confidence_text_class"] = "text-green-600"
            elif top["confidence"] >= 60:
                context["confidence_label"]      = "Moderate Confidence"
                context["confidence_bar_class"]  = "bg-yellow-400"
                context["confidence_text_class"] = "text-yellow-600"
            else:
                context["confidence_label"]      = "Low Confidence"
                context["confidence_bar_class"]  = "bg-red-500"
                context["confidence_text_class"] = "text-red-600"

            #  Risk badge + icon classes decided in Python
            risk_color = top["risk_color"]
            if risk_color == "red":
                context["risk_badge_class"] = "bg-red-100 text-red-600"
                context["risk_icon_class"]  = "bg-red-50 text-red-500"
            elif risk_color == "yellow":
                context["risk_badge_class"] = "bg-yellow-100 text-yellow-600"
                context["risk_icon_class"]  = "bg-yellow-50 text-yellow-500"
            else:
                context["risk_badge_class"] = "bg-green-100 text-green-600"
                context["risk_icon_class"]  = "bg-green-50 text-green-500"

            # Low confidence flag
            context["low_confidence"] = top["confidence"] < 60

        except Exception as e:
            context["error"] = f"Error processing image: {str(e)}"

    return render(request, 'skin-detection.html', context)

def stress_analyser(request):
    return render(request,"stress-analyser.html")

def signup(request):
    return render(request,"signup.html")

def get_started(request):
    return render(request,"get-started.html")

def health_tips(request):
    return render(request,"health-tips.html")