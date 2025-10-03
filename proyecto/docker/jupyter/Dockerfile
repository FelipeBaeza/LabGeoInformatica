FROM jupyter/scipy-notebook:latest

USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Copiar requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Configurar Jupyter
RUN jupyter lab build

WORKDIR /home/jovyan/work
