from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.
class Usuario(AbstractUser):
    is_admin=models.BooleanField(default=False)

    def __str__(self):
        return "{}".format(self.username)

class Diagnostico(models.Model):
    ruta_Imagen=models.CharField(max_length=50)
    ruta_ImagenSegmentada=models.CharField(max_length=50)
    clase=models.CharField(max_length=50)
    descripcion=models.CharField(max_length=200)
    clase_Correccion=models.CharField(max_length=50)
    descripcion_Correccion=models.CharField(max_length=200)
    aprobado=models.BooleanField(default=False)
    updated_at=models.DateField(auto_now=True)
    created_at=models.DateField(auto_now=True)
    usuario=models.ForeignKey(Usuario,null=False,blank=False, on_delete=models.CASCADE)
    