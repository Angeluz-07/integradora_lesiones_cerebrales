from django.conf import settings

import SimpleITK as sitk
import scipy
import joblib
import os

def volume(img: sitk.Image):
  array = sitk.GetArrayFromImage(img)
  pixel_volume = img.GetSpacing()[0] * img.GetSpacing()[1] * img.GetSpacing()[2]
  return (array > 0).sum() * pixel_volume

def max_(img: sitk.Image):
  f = sitk.StatisticsImageFilter()
  f.Execute(img)
  return f.GetMaximum()

def variance(img: sitk.Image):
  f = sitk.StatisticsImageFilter()
  f.Execute(img)
  return f.GetVariance()

def MorphologicalWatershed_(img: sitk.Image):
  transformed = sitk.MorphologicalWatershed(img)
  array = sitk.GetArrayFromImage(transformed)

  f = sitk.StatisticsImageFilter()
  f.Execute(transformed)
  return f.GetSum()

def Z_(img: sitk.Image):
  array = sitk.GetArrayFromImage(img)
  coords = scipy.ndimage.measurements.center_of_mass(array)
  return coords[0]

def X_(img: sitk.Image):
  array = sitk.GetArrayFromImage(img)
  coords = scipy.ndimage.measurements.center_of_mass(array)
  return coords[1]

def Y_(img: sitk.Image):
  array = sitk.GetArrayFromImage(img)
  coords = scipy.ndimage.measurements.center_of_mass(array)
  return coords[2]

def get_feature_vector(img : sitk.Image):
  fn = [
    volume,
    max_,
    variance,
    MorphologicalWatershed_,
    Z_,
    X_,
    Y_
  ]
  return [f(img) for f in fn]

mni152_T1_path = os.path.join(
    settings.BRAIN_TEMPLATES_DIR, 
    'mni_icbm152_t1_tal_nlin_sym_09a.nii'
)
mni152_T1 = sitk.ReadImage(mni152_T1_path, sitk.sitkFloat32)

def preprocess(xpath, ypath):
    ximg = sitk.ReadImage(xpath, sitk.sitkFloat32)
    ximg = sitk.HistogramMatching(ximg, mni152_T1)
    yimg = sitk.ReadImage(ypath, sitk.sitkFloat32)
    yimg = sitk.Divide(yimg, 255.0) # because mask comes with values in {255,0}
    
    masked = sitk.Multiply(ximg, yimg)
    feature_row = get_feature_vector(masked)
    return feature_row

def probs(model, feature_row):
    raw_pred = model.predict_proba([feature_row])
    return {
        'Lacunar' : raw_pred[0][0] ,
        'MCA' : raw_pred[0][1]
    }

def probs_formatted(model, feature_row):
  raw_pred = model.predict_proba([feature_row])
  result = f"""
  Lacunar con una probabilidad de {raw_pred[0][0] * 100}% ,
  MCA con una probabilidad de {raw_pred[0][1] * 100}%
  """
  return result

def predicted_class(model, feature_row):
    raw_pred = model.predict([feature_row])
    CLASSES =  {
        0 : 'Lacunar',
        1 : 'MCA'
    }
    return CLASSES[raw_pred[0]]


model_path = os.path.join(
    settings.MODELS_DIR, 
    os.environ.get('CLASSIFICATION_MAIN_MODEL')
)

model = joblib.load(model_path)

