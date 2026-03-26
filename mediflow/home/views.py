from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def disease_predictor(request):
    return render(request,"disease-predictor.html")

def skin_disease_predictor(request):
    return render(request,'skin-detection.html')

def stress_analyser(request):
    return render(request,"stress-analyser.html")

def signup(request):
    return render(request,"signup.html")

def get_started(request):
    return render(request,"get-started.html")

def health_tips(request):
    return render(request,"health-tips.html")