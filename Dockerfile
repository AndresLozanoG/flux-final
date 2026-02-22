# Imagen base oficial con PyTorch 2.4.0 y CUDA 12.4
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Actualizar pip e instalar librerías sincronizadas
RUN pip install --upgrade pip && \
    pip install runpod==1.1.6 \
    diffusers==0.30.2 \
    transformers==4.44.2 \
    accelerate==0.34.2 \
    sentencepiece==0.2.0 \
    huggingface_hub==0.24.6 \
    protobuf==5.28.0 \
    pillow==10.4.0 \
    peft==0.12.0

# Copiar el handler
COPY handler.py /handler.py

# Iniciar el proceso
CMD ["python", "-u", "/handler.py"]
