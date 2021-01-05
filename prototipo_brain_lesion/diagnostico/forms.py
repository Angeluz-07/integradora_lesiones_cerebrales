from diagnostico.models import Diagnostico
from django import forms

class DiagnosticoForm(forms.ModelForm):
    
    class Meta:
        model = Diagnostico

        fields = [
            'ruta_Imagen',
            'ruta_ImagenSegmentada',
            'clase',
            'descripcion',
            'clase_Correccion',
            'descripcion_Correccion',
            'aprobado',
            'usuario',
        ]

        widgets = {
            'ruta_Imagen':forms.TextInput(),
            'ruta_ImagenSegmentada':forms.TextInput(),
            'clase':forms.TextInput(),
            'descripcion':forms.Textarea(),
            'clase_Correccion':forms.TextInput(),
            'descripcion_Correccion':forms.Textarea(),
            'aprobado':forms.CheckboxInput(),
            'usuario':forms.Select(),
        }  
             