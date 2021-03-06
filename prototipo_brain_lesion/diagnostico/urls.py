from django.urls import path
from diagnostico.views import diagnostic, serve_file, preload_file, save_diagnostic, ListarDiagnostico,  diagnostic_only_user_list, ListarUsuario, CreateUsuario, update_usuario, diagnostic_read
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import TemplateView 
urlpatterns = [  
    path('home/',TemplateView.as_view(template_name='home.html'), name="home"),
    path('diagnostic/<str:type_>',diagnostic, name='diagnostic'),
    path('diagnostic/read/<str:id>',diagnostic_read, name='diagnostic_read'),
    path('save/',save_diagnostic,name='save_diagnostic'),
    path('listar/',ListarDiagnostico.as_view(),name='listDiagnostic'),
    path('file/',preload_file, name='preload_file'),
    path('file/<str:file_name>',serve_file, name='serve_file'),
    path('listar/usuario/<int:pk>',diagnostic_only_user_list, name='list_only_user'),
    path('usuarios/',ListarUsuario.as_view(), name='listUser'),
    path('usuarios/create',CreateUsuario.as_view(), name='create_user'),
    path('usuarios/update/<int:id>',update_usuario, name='update_user'),
]
