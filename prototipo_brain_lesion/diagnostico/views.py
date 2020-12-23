from django.shortcuts import render,redirect
from django.conf import settings
from django.http import FileResponse
from django.core.files.storage import FileSystemStorage
import os

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
            'cargarMRI.html',
            context = {
                'original': fileMRI.name,
                'mask': url_mri_mask
            }
        )
    else:
        return render(request, 'cargarMRI.html')
