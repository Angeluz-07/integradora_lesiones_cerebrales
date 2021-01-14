from django.shortcuts import render,redirect
from django.urls import reverse_lazy
from django.conf import settings
from django.http import FileResponse
from django.core.files.storage import FileSystemStorage
from diagnostico.forms import DiagnosticoForm, UsuarioForm
from diagnostico.models import Usuario, Diagnostico
import os
from django.views.generic import ListView, CreateView, UpdateView
from diagnostico.segmentation import model as segmentation_model
from diagnostico.segmentation import preprocess_ximg, postprocess_pred
from diagnostico.classification import model as classification_model
from diagnostico.classification import preprocess as classification_preprocess
from diagnostico.classification import probs, predicted_class

import SimpleITK as sitk

def serve_file(request, file_name):
    path = os.path.join(settings.MEDIA_ROOT, file_name)
    response = FileResponse(open(path, 'rb'))
    return response

def remove_file_if_exists(file_name):
    file_path = os.path.join(settings.MEDIA_ROOT,file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

def generate_mask(xpath, output_path):
    print("[1] Leyendo imagen...")
    ximg = sitk.ReadImage(xpath, sitk.sitkFloat32)
    print("[2] Preprocesando imagen...")
    x3d =  preprocess_ximg(ximg)
    print("[3] Model.predict()...")
    raw_pred = segmentation_model.predict(x3d,batch_size=8,verbose=1)
    print("[4] Postprocesando prediccion...")
    output = postprocess_pred(raw_pred, xpath)
    print("[5] Guardando segmento...")
    sitk.WriteImage(output, output_path)


def diagnostic(request, type_:str):
    if type_ == 'new' and request.method == 'POST' and request.FILES['fileMRI']:
        fileMRI = request.FILES['fileMRI']
        mri_file_name = fileMRI.name

        print('Guardando archivo...')
        remove_file_if_exists(mri_file_name)
        FileSystemStorage().save(mri_file_name, fileMRI)
        print('Archivo guardado :', f'{mri_file_name}')

        print("Segmentando Lesion...")
        mri_file_path = os.path.join(settings.MEDIA_ROOT, mri_file_name)
        mask_file_name = mri_file_name.split('.')[0] +'_maskGenerated' + '.nii.gz'
        mask_file_path = os.path.join(settings.MEDIA_ROOT, mask_file_name)
        #generate_mask(mri_file_path, mask_file_path)
        print("Segmento generado : ", f'{mask_file_name}')

        print("Clasificando Lesion...")
        feature_row = classification_preprocess(mri_file_path,mask_file_path)
        _probs = probs(classification_model, feature_row)
        _predicted_class = predicted_class(classification_model, feature_row)
        print("Classificacion generada : ", _probs, _predicted_class)

        context = {
            'original': mri_file_name,
            'mask': mask_file_name,
            'clase_pred': _predicted_class,
            'descripcion': _probs,
        }

        request.session['diagnostic_values'] = context
        return render(request,'diagnostico.html',context=context)
    elif type_ == 'update':
        context = request.session['diagnostic_values']
        return render(request,'diagnostico.html',context=context)
    else:
        return render(request, 'home.html')

def save_diagnostic(request):
    if request.method == 'POST':
        form=DiagnosticoForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('diagnostic',type_='new')
        else:
            form=DiagnosticoForm()           
    return render(request,'diagnostico.html',{'form':form})

class ListarDiagnostico(ListView):
    model = Diagnostico
    template_name = "listarDiagnostico.html"
    context_object_name = 'diagnosticos'
    queryset = Diagnostico.objects.order_by('id')

class ListarUsuario(ListView):
    model = Usuario
    template_name = "listarUsuario.html"
    context_object_name = "usuarios"
    queryset = Usuario.objects.order_by('id')


class CreateUsuario(CreateView):
    model = Usuario
    template_name = "crearUsuario.html"
    form_class = UsuarioForm
    success_url = reverse_lazy('listUser')

class UpdateUsuario(UpdateView):
    model = Usuario
    template_name = "crearUsuario.html"
    form_class = UsuarioForm
    success_url = reverse_lazy('listUser') 