from diagnostico.models import Diagnostico
from django import forms

class DiagnosticoForm(forms.ModelForm):
    
    class Meta:
        model = Diagnostico

        fields = [
            'nombre_mri',
            'nombre_mask',
            'clase_predicha',
            'clase_correccion',
            'descripcion',
            'aprobado',
            'usuario'
        ]

        widgets = {
            'nombre_mri':forms.TextInput(),
            'nombre_mask':forms.TextInput(),
            'clase_predicha':forms.TextInput(),
            'clase_correccion':forms.TextInput(),
            'descripcion':forms.Textarea(),
            'aprobado':forms.CheckboxInput(),
            'usuario':forms.Select()
        }       