from django.shortcuts import render,redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
# Create your views here.

def upload_file(request):
    
    if request.method == 'POST' and request.FILES['fileMRI']:
        print('entro')
        fileMRI = request.FILES['fileMRI']
        if os.path.exists(os.path.join(settings.MEDIA_ROOT,fileMRI.name)):
            os.remove(os.path.join(settings.MEDIA_ROOT,fileMRI.name)) 
        fs = FileSystemStorage()   
        filename = fs.save(fileMRI.name,fileMRI)
        file_url = fs.url(filename)
        xpath=os.path.join(settings.MEDIA_ROOT,fileMRI.name)
        
        return render(request, 'cargarMRI.html') 
    else:
        return render(request, 'cargarMRI.html')      