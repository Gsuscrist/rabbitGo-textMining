# Usar una imagen base de Python
FROM python:latest

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos y el código fuente al contenedor
COPY requirements.txt requirements.txt
COPY api.py api.py

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "api.py"]
