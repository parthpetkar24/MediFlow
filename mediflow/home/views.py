import os
import sys
import json
import numpy as np
import importlib.util
import torch
import joblib
from torchvision import transforms
from PIL import Image
from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST
from skin_model.model import SkinCNN
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import UserProfile, SkinDiseaseHistory, DiseaseHistory, StressHistory, Disease_Info, Skin_Disease_Info

# ── Absolute paths anchored to the home app directory ──────────────────────────
APP_DIR     = os.path.dirname(os.path.abspath(__file__))   # → .../home/
PROJECT_DIR = os.path.dirname(APP_DIR) 

DISEASE_MODEL_PATH   = os.path.join(PROJECT_DIR, 'disease_predictor', 'disease_model.pkl')
SYMPTOM_ENCODER_PATH = os.path.join(PROJECT_DIR, 'disease_predictor', 'symptom_encoder.pkl')
DISEASE_ENCODER_PATH = os.path.join(PROJECT_DIR, 'disease_predictor', 'disease_encoder.pkl')

disease_model    = joblib.load(DISEASE_MODEL_PATH)
symptom_encoder  = joblib.load(SYMPTOM_ENCODER_PATH)
disease_encoder  = joblib.load(DISEASE_ENCODER_PATH)

VALID_SYMPTOMS = set(symptom_encoder.classes_)

# ── Skin model — add the home app dir to path so "skin_model" resolves cleanly ─
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


MODEL_PATH  = os.path.join(PROJECT_DIR, 'skin_model', 'skin_model.pth')
LABELS_PATH = os.path.join(PROJECT_DIR, 'skin_model', 'labels.joblib')

FULL_NAMES = {
    'nv' : 'Melanocytic Nevi (Moles)',
    'mel': 'Melanoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'ak' : 'Actinic Keratosis',
}

RISK_LEVEL = {
    'nv' : ('Low',    'green'),
    'mel': ('High',   'red'),
    'bcc': ('High',   'red'),
    'bkl': ('Low',    'green'),
    'ak' : ('Medium', 'yellow'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = joblib.load(LABELS_PATH)
skin_model = SkinCNN(len(labels))
skin_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
skin_model.to(device)
skin_model.eval()
 
 
# ── Stress model ───────────────────────────────────────────────────────────────
STRESS_MODEL_DIR       = os.path.join(PROJECT_DIR, 'stress_analyzer')
STRESS_ANALYZER_PY     = os.path.join(STRESS_MODEL_DIR, 'stress_analyzer.py')
STRESS_VECTORIZER_PATH = os.path.join(STRESS_MODEL_DIR, 'stress_model','stress_vectorizer.pkl')
STRESS_MODEL_PATH_FILE = os.path.join(STRESS_MODEL_DIR, 'stress_model','stress_model.pkl')
 
_spec   = importlib.util.spec_from_file_location("stress_analyzer", STRESS_ANALYZER_PY)
_sa_mod = importlib.util.module_from_spec(_spec)
 
sys.modules["stress_analyzer"] = _sa_mod
_real_main = sys.modules.get("__main__")
sys.modules["__main__"] = _sa_mod         
 
_spec.loader.exec_module(_sa_mod)          
 
stress_vectorizer = joblib.load(STRESS_VECTORIZER_PATH)
stress_clf        = joblib.load(STRESS_MODEL_PATH_FILE)

if _real_main is not None:
    sys.modules["__main__"] = _real_main
 
# Grab the inference helper.
predict_from_text = _sa_mod.predict_from_text
 
# Human-readable label 
STRESS_LABEL_DISPLAY = {
    "no_stress":       "No Stress",
    "coping":          "Coping",
    "moderate_stress": "Moderate Stress",
    "high_stress":     "High Stress",
    "burnout":         "Burnout",
}
 
STRESS_LABEL_COLOR = {
    "no_stress":       "#34d399",
    "coping":          "#38bdf8",
    "moderate_stress": "#fbbf24",
    "high_stress":     "#fb923c",
    "burnout":         "#f43f5e",
}

# ── Skin prediction helper ──────────────────────────────────────────────────────
def predict_skin(image):
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
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
            'label'     : lbl,
            'full_name' : FULL_NAMES.get(lbl, lbl),
            'confidence': conf,
            'risk'      : RISK_LEVEL.get(lbl, ('Unknown', 'gray'))[0],
            'risk_color': RISK_LEVEL.get(lbl, ('Unknown', 'gray'))[1],
        })
    return results


# ── Views ───────────────────────────────────────────────────────────────────────
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def get_started(request):
    return render(request, 'get-started.html')

def signup(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        # ── Pull fields (matching the new signup.html hidden inputs) ──
        first_name    = request.POST.get('first_name',    '').strip()
        last_name     = request.POST.get('last_name',     '').strip()
        email         = request.POST.get('email',         '').strip()

        # … rest of signup logic unchanged …

    return render(request, 'signup.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        email    = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()

        if not email or not password:
            messages.error(request, 'Email and password are required.')
            return render(request, 'login.html')

        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.first_name or user.email}!')
            next_url = request.GET.get('next', 'index')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid email or password.')

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('index')

def health_tips(request):
    return render(request, 'health-tips.html')

def stress_analyser(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            user_text = body.get('text', '').strip()
        except (json.JSONDecodeError, AttributeError):
            user_text = request.POST.get('text', '').strip()
 
        if not user_text:
            return JsonResponse({'error': 'Please enter some text.'}, status=400)
 
        result = predict_from_text(user_text, stress_vectorizer, stress_clf)
        if request.user.is_authenticated:
            StressHistory.objects.create(
            user=request.user,
            input_text=user_text,
            label=result['label'],
            label_display=STRESS_LABEL_DISPLAY.get(result['label'], result['label']),
            score=result['score'],
            advice=result.get('advice', ''),
            color=STRESS_LABEL_COLOR.get(result['label'], '#94a3b8'),
            probabilities=result.get('probabilities', {}),
        )
 
        return JsonResponse({
            'label':         result['label'],
            'label_display': STRESS_LABEL_DISPLAY.get(result['label'], result['label']),
            'score':         result['score'],
            'color':         STRESS_LABEL_COLOR.get(result['label'], '#94a3b8'),
            'advice':        result['advice'],
            'probabilities': result['probabilities'],
        })
 
    return render(request, 'stress-analyser.html')

def disease_predictor(request):
    context = {'all_symptoms': sorted(VALID_SYMPTOMS)}
    
    if request.method == 'POST':
        symptoms = request.POST.getlist('symptoms')
        # Strip out anything not in the training vocabulary
        symptoms = [s for s in symptoms if s in VALID_SYMPTOMS]
        context['selected'] = symptoms

        if len(symptoms) < 2:
            context['need_more'] = True
            return render(request, 'disease-predictor.html', context)

        try:
            encoded       = symptom_encoder.transform([symptoms])
            prediction    = disease_model.predict(encoded)
            probabilities = disease_model.predict_proba(encoded)
            disease       = disease_encoder.inverse_transform(prediction)[0]
            confidence    = round(float(np.max(probabilities)) * 100, 2)

            top3 = np.argsort(probabilities[0])[-3:][::-1]
            top_diseases = [
                {
                    'name'      : disease_encoder.inverse_transform([i])[0],
                    'confidence': round(float(probabilities[0][i]) * 100, 2),
                }
                for i in top3
            ]

            context.update({
                'predicted_disease': disease,
                'confidence'       : confidence,
                'top_diseases'     : top_diseases,
            })

            # ── Fetch description & remedies for the predicted disease ──────────
            try:
                disease_info = Disease_Info.objects.get(name__iexact=disease)
                context['disease_info'] = {
                    'description': disease_info.description,
                    'medications': [
                        m for m in [
                            disease_info.medication1,
                            disease_info.medication2,
                            disease_info.medication3,
                        ] if m
                    ],
                }
            except Disease_Info.DoesNotExist:
                context['disease_info'] = None

            if request.user.is_authenticated:
                DiseaseHistory.objects.create(
                user=request.user,
                symptoms=symptoms,
                predicted_disease=disease,
                confidence=confidence,
                top_diseases=top_diseases,
                )
        except Exception as e:
            context['need_more'] = True

    return render(request, 'disease-predictor.html', context)


def skin_disease_predictor(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('skin_image'):
        try:
            img_file = request.FILES['skin_image']
            image    = Image.open(img_file).convert('RGB')
            results  = predict_skin(image)
            top      = results[0]

            context['results']   = results
            if request.user.is_authenticated:
                SkinDiseaseHistory.objects.create(
                user=request.user,
                predicted_label=top['label'],
                full_name=top['full_name'],
                confidence=top['confidence'],
                risk_level=top['risk'],
                top_results=results,
            )
            context['top']       = top
            context['bar_width'] = int(top['confidence'])

            if top['confidence'] >= 80:
                context['confidence_label']      = 'High Confidence'
                context['confidence_bar_class']  = 'bg-green-500'
                context['confidence_text_class'] = 'text-green-600'
            elif top['confidence'] >= 60:
                context['confidence_label']      = 'Moderate Confidence'
                context['confidence_bar_class']  = 'bg-yellow-400'
                context['confidence_text_class'] = 'text-yellow-600'
            else:
                context['confidence_label']      = 'Low Confidence'
                context['confidence_bar_class']  = 'bg-red-500'
                context['confidence_text_class'] = 'text-red-600'

            risk_color = top['risk_color']
            if risk_color == 'red':
                context['risk_badge_class'] = 'bg-red-100 text-red-600'
                context['risk_icon_class']  = 'bg-red-50 text-red-500'
            elif risk_color == 'yellow':
                context['risk_badge_class'] = 'bg-yellow-100 text-yellow-600'
                context['risk_icon_class']  = 'bg-yellow-50 text-yellow-500'
            else:
                context['risk_badge_class'] = 'bg-green-100 text-green-600'
                context['risk_icon_class']  = 'bg-green-50 text-green-500'

            context['low_confidence'] = top['confidence'] < 60

            # ── Fetch description & remedy for the predicted skin disease ────────
            try:
                skin_info = Skin_Disease_Info.objects.get(name__iexact=top['full_name'])
                context['skin_info'] = {
                    'description': skin_info.description,
                    'medication' : skin_info.medication1,
                }
            except Skin_Disease_Info.DoesNotExist:
                context['skin_info'] = None

        except Exception as e:
            context['error'] = f'Error processing image: {str(e)}'

    return render(request, 'skin-detection.html', context)

@login_required(login_url='/login/')
def history_overview(request):
    context = {
        'skin_count':    SkinDiseaseHistory.objects.filter(user=request.user).count(),
        'disease_count': DiseaseHistory.objects.filter(user=request.user).count(),
        'stress_count':  StressHistory.objects.filter(user=request.user).count(),
        'recent_skin':   SkinDiseaseHistory.objects.filter(user=request.user)[:3],
        'recent_disease':DiseaseHistory.objects.filter(user=request.user)[:3],
        'recent_stress': StressHistory.objects.filter(user=request.user)[:3],
    }
    return render(request, 'overview.html', context)


@login_required(login_url='/login/')
def skin_history(request):
    records = SkinDiseaseHistory.objects.filter(user=request.user)
    return render(request, 'skin_history.html', {'records': records})


@login_required(login_url='/login/')
def disease_history_view(request):
    records = DiseaseHistory.objects.filter(user=request.user)
    return render(request, 'disease_history.html', {'records': records})


@login_required(login_url='/login/')
def stress_history_view(request):
    records = StressHistory.objects.filter(user=request.user)
    return render(request, 'stress_history.html', {'records': records})