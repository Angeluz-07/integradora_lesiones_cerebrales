import ants
import SimpleITK as sitk
import numpy as np
import os
from django.conf import settings


mni_T1_path = os.path.join(
    settings.BRAIN_TEMPLATES_DIR, 
    'mni_icbm152_t1_tal_nlin_sym_09a.nii'
)

mni_T1_ANTSPY = ants.image_read(mni_T1_path)
mni_T1_SITK = sitk.ReadImage(mni_T1_path, sitk.sitkFloat32)

def normalize(file_dir, file_name):
  print("[1] Normalización por histograma...")
  raw_file_path = file_dir + '/' + file_name.split('.')[0] +'.'+ 'nii.gz'
  x3d = sitk.ReadImage(raw_file_path, sitk.sitkFloat32)
  histogramNormalized = sitk.HistogramMatching(x3d, mni_T1_SITK)

  print("[2] Guardando imagen normalizada por histograma...")
  histogramNormalized_file_path = file_dir + '/' + file_name.split('.')[0] + '_histNormalized.' + 'nii.gz'
  sitk.WriteImage(histogramNormalized,histogramNormalized_file_path)

  print("[3] Inicio de registro a MNI-152...")
  x3d = ants.image_read(histogramNormalized_file_path)
  transformed = ants.registration(
      fixed=mni_T1_ANTSPY, 
      moving=x3d, 
      type_of_transform='SyN',
      verbose=True
  )

  print("[4] Guardando imagen registrada a MNI-152...")
  normalized_file_path = file_dir + '/' + file_name.split('.')[0] + '_normalized.' + 'nii.gz'
  transformed['warpedmovout'].to_file(normalized_file_path)

  new_mri_file_name =  file_name.split('.')[0] + '_normalized.' + 'nii.gz'
  return new_mri_file_name
