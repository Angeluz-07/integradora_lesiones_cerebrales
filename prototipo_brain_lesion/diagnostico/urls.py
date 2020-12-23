from django.urls import path
from .views import upload_file, serve_file
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import TemplateView 
urlpatterns = [  
    path('home/',TemplateView.as_view(template_name='home.html'),name="home"),
    path('upload/',upload_file, name='upload'),
    path('file/<str:file_name>',serve_file, name='serve_file'),
]