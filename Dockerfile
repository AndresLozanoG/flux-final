# Imagen de NVIDIA de última generación con soporte para todas las GPUs modernas
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos solo las librerías de IA. PyTorch ya viene optimizado en la base.
RUN pip install --upgrade pip && \
    pip install runpod diffusers transformers accelerate sentencepiece huggingface_hub protobuf pillow peft

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
