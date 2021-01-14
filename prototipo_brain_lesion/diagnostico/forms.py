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
            'username':forms.TextInput(attrs={'class':'form-control'}),
            'first_name':forms.TextInput(attrs={'class':'form-control'}),
            'last_name':forms.TextInput(attrs={'class':'form-control'}),
            'password':forms.TextInput(attrs={'class':'form-control'}),
            'is_admin':forms.CheckboxInput(attrs={'class':'form-check'}),
        }

    def save(self, commit=True):
        # Save the provided password in hashed format
        user = super(UsuarioForm, self).save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user    
             