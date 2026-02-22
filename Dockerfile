# Imagen base oficial con PyTorch 2.4.0 (Esta soluciona el error de XPU)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Actualizar pip e instalar librerías sin forzar versiones estrictas
# Esto permite que pip encuentre la combinación perfecta para esta imagen
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    huggingface_hub \
    protobuf \
    pillow \
    peft

# Copiar el handler
COPY handler.py /handler.py

# Iniciar el proceso
CMD ["python", "-u", "/handler.py"]
