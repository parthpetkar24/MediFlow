from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user       = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    avatar     = models.ImageField(upload_to='avatars/', blank=True, null=True)
    phone      = models.CharField(max_length=15, blank=True, null=True)
    date_of_birth = models.DateField(blank=True, null=True)
    gender     = models.CharField(max_length=10, blank=True, null=True,
                    choices=[('male','Male'), ('female','Female'), ('other','Other')])
    blood_group = models.CharField(max_length=5, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.get_full_name()} — Profile"

    class Meta:
        db_table = "user_profile"


class SkinDiseaseHistory(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, related_name='skin_history')
    image           = models.ImageField(upload_to='skin_history/', blank=True, null=True)
    predicted_label = models.CharField(max_length=100)
    full_name       = models.CharField(max_length=200)
    confidence      = models.FloatField()
    risk_level      = models.CharField(max_length=20)   # Low / Medium / High
    top_results     = models.JSONField(default=list)     # stores top-3 predictions
    created_at      = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} — {self.full_name} ({self.confidence}%)"

    class Meta:
        db_table  = "skin_disease_history"
        ordering  = ['-created_at']


class DiseaseHistory(models.Model):
    user             = models.ForeignKey(User, on_delete=models.CASCADE, related_name='disease_history')
    symptoms         = models.JSONField(default=list)        # list of symptom strings
    predicted_disease = models.CharField(max_length=200)
    confidence       = models.FloatField()
    top_diseases     = models.JSONField(default=list)        # top-3 [{name, confidence}]
    created_at       = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} — {self.predicted_disease} ({self.confidence}%)"

    class Meta:
        db_table = "disease_history"
        ordering = ['-created_at']


class StressHistory(models.Model):
    STRESS_LABELS = [
        ('no_stress',       'No Stress'),
        ('coping',          'Coping'),
        ('moderate_stress', 'Moderate Stress'),
        ('high_stress',     'High Stress'),
        ('burnout',         'Burnout'),
    ]

    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name='stress_history')
    input_text    = models.TextField()
    label         = models.CharField(max_length=30, choices=STRESS_LABELS)
    label_display = models.CharField(max_length=50)
    score         = models.FloatField()
    advice        = models.TextField(blank=True, null=True)
    color         = models.CharField(max_length=10, default='#94a3b8')
    probabilities = models.JSONField(default=dict)
    created_at    = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} — {self.label_display} ({self.score})"

    class Meta:
        db_table = "stress_history"
        ordering = ['-created_at']


class Disease_Info(models.Model):
    name=models.CharField(max_length=30)
    description=models.TextField()
    medication1=models.CharField(max_length=200)
    medication2=models.CharField(max_length=200)
    medication3=models.CharField(max_length=200)

    def __str__(self):
        return self.name
    
class Skin_Disease_Info(models.Model):
    name=models.CharField(max_length=30)
    description=models.TextField()
    medication1=models.CharField(max_length=200)

    def __str__(self):
        return self.name
