from django.urls import path
from diagnostico.views import generate_diagnostic, serve_file, save_diagnostic, new_diagnostic, new_save_diagnostic, ListarDiagnostico
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import TemplateView 
urlpatterns = [  
    path('home/',TemplateView.as_view(template_name='home.html'), name="home"),
    path('generate-diagnostic/',generate_diagnostic, name='generate_diagnostic'),
    path('save/',save_diagnostic,name='save_diagnostic'),
    path('new/<str:nombre_mri>/<str:nombre_mask>/<str:clase_pred>/<str:descripcion>/',new_diagnostic,name='new_diagnostic'),
    path('new_save/',new_save_diagnostic,name='new_save_diagnostic'),
    path('listar/',ListarDiagnostico.as_view(),name='listDiagnostic'),
    path('file/<str:file_name>/',serve_file, name='serve_file'),
]