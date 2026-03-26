from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('index/',views.index,name='index'),
    path('about/',views.about,name='about'),
    path('disease_predictor/',views.disease_predictor,name='disease_predictor'),
    path('skin_disease_predictor/',views.skin_disease_predictor,name='skin_disease_predictor'),
    path('stress_analyser/',views.stress_analyser,name='stress_analyser'),
    path('signup/',views.signup,name='signup'),
    path('health_tips/',views.health_tips,name='health_tips'),
    path('get_started/',views.get_started,name='get_started'),
]