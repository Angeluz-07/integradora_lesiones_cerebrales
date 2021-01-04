from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.
class Usuario(AbstractUser):
    is_admin=models.BooleanField(default=False)

    def __str__(self):
        return "{}".format(self.username)

class Diagnostico(models.Model):
    nombre_mri=models.CharField(max_length=50)
    nombre_mask=models.CharField(max_length=50)
    clase_predicha=models.CharField(max_length=50)
    clase_correccion=models.CharField(max_length=50)
    descripcion=models.CharField(null=True,blank=True,max_length=200)
    aprobado=models.BooleanField(default=False)
    fecha=models.DateField(auto_now=True)
    usuario=models.ForeignKey(Usuario,null=False,blank=False, on_delete=models.CASCADE)
    

    

    
        
    
            