from django.contrib import admin
from .models import Disease_Info,Skin_Disease_Info,UserProfile

admin.site.site_header = "Mediflow Administration"
admin.site.site_title = "Mediflow Admin Portal"
admin.site.index_title = "Welcome to Mediflow Dashboard"

# Register your models here.
admin.site.register(Disease_Info)
admin.site.register(Skin_Disease_Info)
admin.site.register(UserProfile)