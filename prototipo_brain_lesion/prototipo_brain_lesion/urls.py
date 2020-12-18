"""prototipo_brain_lesion URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth.views import LoginView, TemplateView, logout_then_login 
from django.urls import path, include
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',TemplateView.as_view(template_name='base.html'),name="base"),
    path('accounts/login/',LoginView.as_view(template_name='login.html'),name="login"),
    path('home/',login_required(TemplateView.as_view(template_name='home.html')),name="home"),
    path('logout/', logout_then_login,name='logout')
]
