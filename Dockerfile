FROM nvcr.io/nvidia/pytorch:24.08-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --upgrade pip && \
    pip install runpod diffusers transformers accelerate sentencepiece huggingface_hub protobuf pillow peft
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
