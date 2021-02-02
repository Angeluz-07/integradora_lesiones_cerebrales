import ants
import os
from django.conf import settings


mni_T1_path = os.path.join(
    settings.BRAIN_TEMPLATES_DIR, 
    'mni_icbm152_t1_tal_nlin_sym_09a.nii'
)

mni_T1_ANTSPY = ants.image_read(mni_T1_path)

def register(file_dir, file_name):
  print("[1] Inicio de registro a MNI-152...")
  raw_file_path = file_dir + '/' + file_name.split('.')[0] +'.nii.gz'
  x3d = ants.image_read(raw_file_path)
  transformed = ants.registration(
      fixed=mni_T1_ANTSPY, 
      moving=x3d, 
      type_of_transform='SyN',
      verbose=True
  )

  print("[2] Guardando imagen registrada a MNI-152...")
  registered_file_path = file_dir + '/' + file_name.split('.')[0] + '_registered.nii.gz'
  transformed['warpedmovout'].to_file(registered_file_path)

  new_mri_file_name =  file_name.split('.')[0] + '_registered.nii.gz'
  return new_mri_file_name
