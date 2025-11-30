# -------------------------------------------------
# Imagen base ligera
# -------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------
# Variables de entorno  
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# -------------------------------------------------
# Working folder inside the container.
# Create a directory called /app inside the container. 
#(Image vs. Container): The Dockerfile defines the image. 
#When you run the image (with docker run), a container is created, and /app is a directory within that container.
WORKDIR /app

# -------------------------------------------------
# Dependencias del sistema necesarias para librerías como OpenCV/PIL
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Copiar requirements y instalar dependencias Python
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 50 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --retries 50 -r requirements.txt && \
    pip install --force-reinstall --no-deps numpy==1.26.4
    #pip install --no-cache-dir -r requirements.txt

#We copy all the content inside the "project" folder and
#put or paste it inside the directory created in our 
#container that is created when running our image with "docker run"
#That is, /app
#COPY project/ ./
COPY python/malaria_clasification/src/ ./
COPY python/malaria_clasification/models/ ./models/
#COPY models/ ./models/
# test_images se deja solo para local, no producción

# -------------------------------------------------
# Exponer puerto
# -------------------------------------------------
EXPOSE ${PORT}
EXPOSE 7860

# -------------------------------------------------
# Comando para levantar la API con Uvicorn
# -------------------------------------------------

#CMD ["sh", "-c", "uvicorn inference.main:app --host 0.0.0.0 --port ${PORT} & python ui/gradio_app.py & wait"]
CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
