from django.urls import path
from .views import upload_file
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import TemplateView 
urlpatterns = [  
    path('home/',login_required(TemplateView.as_view(template_name='home.html')),name="home"), 
    path('upload/',login_required(upload_file),name='upload'),
]