from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('about/',views.about,name='about'),
    path('disease-predictor/',views.disease_predictor,name='disease_predictor'),
    path('skin-detection/',views.skin_disease_predictor, name='skin_disease_predictor'),
    path('stress-analyser/',views.stress_analyser,name='stress_analyser'),
    path('login/',views.login_view,name='login'),
    path('signup/',views.signup,name='signup'),
    path('logout/',views.logout_view,name='logout'),
    path('get-started/',views.get_started,name='get_started'),
    path('health-tips/',views.health_tips,name='health_tips'),
    path('overview/', views.history_overview,name='history_overview'),
    path('skin/', views.skin_history, name='skin_history'),
    path('disease/',views.disease_history_view, name='disease_history'),
    path('stress/',views.stress_history_view, name='stress_history'),
]
