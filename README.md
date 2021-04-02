# Brain Lesion Detector
_Código fuente del proyecto de materia integradora **Detección de lesiones cerebrales mediante análisis de imagenes MRI(Magnetic Resonance Imaging) mediante el uso de deep learning**. Proyecto previo a la obtención del título de Ingeniero en Ciencias Computacionales de la ESPOL(Escuela Superior Politécnica del Litoral) llevado a cabo en el segundo término 2020._

## Correr aplicación (Windows OS)
1. Iniciar el ambiente virtual python.
2. Establecer variables de ambiente :
```
.env.bat
```
3. Cambiar directorio e iniciar aplicación :
```
cd prototipo_brain_lesion
python manage.py runserver
```
## Correr aplicación con docker-compose (Linux)
1. Instalar Docker y Docker Compose.
2. Crear archivo `.env` a partir del archivo `.env.example`.
3. Establecer variables de ambiente :
```bash
source .env
```
4. Levantar servicios:
```bash
sudo docker-compose up -d

sudo docker-compose down # detener servicios
sudo docker-compose ps # inspeccionar servicios
```

5. Acceder a los servicios:
- `http://localhost:8000/` aplicación web.
- `http://localhost:5052/` consola de pgadmin4.

## Requerimientos
- Python >=3.7.1
- PostgreSQL >=11.3
