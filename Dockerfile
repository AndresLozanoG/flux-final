# Imagen base con kernels pre-compilados para RunPod
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos las librerías necesarias. 
# No forzamos versión de torch porque ya viene perfecta en la base.
RUN pip install --upgrade pip && \
    pip install runpod diffusers transformers accelerate sentencepiece huggingface_hub protobuf pillow peft

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
