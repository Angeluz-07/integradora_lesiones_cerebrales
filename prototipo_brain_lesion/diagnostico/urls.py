from django.urls import path
from diagnostico.views import diagnostic, serve_file, preload_file, save_diagnostic, ListarDiagnostico
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import TemplateView 
urlpatterns = [  
    path('home/',TemplateView.as_view(template_name='home.html'), name="home"),
    path('diagnostic/<str:type_>',diagnostic, name='diagnostic'),
    path('save/',save_diagnostic,name='save_diagnostic'),
    path('listar/',ListarDiagnostico.as_view(),name='listDiagnostic'),
    path('file/',preload_file, name='preload_file'),
    path('file/<str:file_name>',serve_file, name='serve_file'),
]