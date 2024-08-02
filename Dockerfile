# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de requerimientos y el script de la aplicación
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY templates/ templates/

# Instala las dependencias
RUN pip install -r requirements.txt

# Expone el puerto en el que correrá la aplicación
EXPOSE 8080

# Comando para correr la aplicación
CMD ["python", "app.py"]
