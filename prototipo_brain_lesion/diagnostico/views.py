from django.shortcuts import render,redirect
from django.conf import settings
from django.http import FileResponse
from django.core.files.storage import FileSystemStorage
from diagnostico.forms import DiagnosticoForm
from diagnostico.models import Usuario, Diagnostico
import os
from django.views.generic import ListView 
from diagnostico.segmentation import model as segmentation_model
from diagnostico.segmentation import preprocess_ximg, postprocess_pred
import SimpleITK as sitk

def serve_file(request, file_name):
    path = os.path.join(settings.MEDIA_ROOT, file_name)
    response = FileResponse(open(path, 'rb'))
    return response

def upload_file(request):
    if request.method == 'POST' and request.FILES['fileMRI']:
        print('Guardando archivo...')
        fileMRI = request.FILES['fileMRI']
        if os.path.exists(os.path.join(settings.MEDIA_ROOT,fileMRI.name)):
            os.remove(os.path.join(settings.MEDIA_ROOT,fileMRI.name)) 
        fs = FileSystemStorage()   
        filename = fs.save(fileMRI.name,fileMRI)
        file_url = fs.url(filename)
        xpath=os.path.join(settings.MEDIA_ROOT,fileMRI.name)
        
        ### Segmentation start
        xpath = xpath
        output_path = os.path.join(settings.MEDIA_ROOT, fileMRI.name.split('.')[0] +'_maskGenerated' + '.nii.gz')
        print(xpath)
        print(output_path)
        print(fs.url(fileMRI.name))
        """
        ximg = sitk.ReadImage(xpath, sitk.sitkFloat32)
        x3d =  preprocess_ximg(ximg)
        raw_pred = segmentation_model.predict(x3d,batch_size=8,verbose=1)
        output = postprocess_pred(raw_pred, xpath)
        sitk.WriteImage(output, output_path)
        """
        url_mri_mask = fileMRI.name.split('.')[0] +'_maskGenerated' + '.nii.gz'
        ### Segmentation end
       
        return render(
            request,
            'diagnostico.html',
            context = {
                'original': fileMRI.name,
                'mask': url_mri_mask,
                'clase_pred':'MCA',
                'descripcion':'Probabilidad MCA 85%\nLacunar 20%\nControl 0%',
            }
        )
    else:
        return render(request, 'cargarMRI.html')

def save_diagnostic(request):
    if request.method == 'POST':
        form=DiagnosticoForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('upload')
        else:
            form=DiagnosticoForm()           
    return render(request,'diagnostico.html',{'form':form})

def new_diagnostic(request,nombre_mri,nombre_mask,clase_pred):
    return render(request,'rechazarMRI.html', context = {
                'original': nombre_mri,
                'mask': nombre_mask,
                'clase_pred': clase_pred,
            })
    
def new_save_diagnostic(request):
    if request.method == 'POST':
        form=DiagnosticoForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('upload')
        else:
            form=DiagnosticoForm()           
    return render(request,'rechazarMRI.html',{'form':form})


class ListarDiagnostico(ListView):
    model = Diagnostico
    template_name = "listarDiagnostico.html"
    context_object_name = 'diagnosticos'
    queryset=Diagnostico.objects.order_by('id')
    
     


    