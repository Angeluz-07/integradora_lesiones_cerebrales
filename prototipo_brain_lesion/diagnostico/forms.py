from diagnostico.models import Diagnostico, Usuario
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

class UsuarioForm(forms.ModelForm):
    
    class Meta:
        model = Usuario
        fields = [
            'username',
            'first_name',
            'last_name',
            'password',
            'is_admin',
        ]

        widgets = {
            'username':forms.TextInput(),
            'first_name':forms.TextInput(),
            'last_name':forms.TextInput(),
            'password':forms.TextInput(),
            'is_admin':forms.CheckboxInput(),
        }
             