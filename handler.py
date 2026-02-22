import runpod
import torch
from diffusers import FluxPipeline
import base64
import os
from huggingface_hub import login
from io import BytesIO

# 1. Autenticación con Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Variable global para el modelo
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Cargando modelo FLUX en la GPU...")
        # Cargamos el modelo
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # OPTIMIZACIÓN DE MEMORIA: Vital para GPUs de 24GB/48GB
        pipe.enable_model_cpu_offload() 
        
        # ANTÍDOTO DEFINITIVO: Forzar el procesador de atención estándar
        # Esto elimina el error 'enable_gqa' de raíz
        pipe.transformer.set_default_attn_processor()
        
        print("Modelo cargado exitosamente.")

def handler(job):
    """
    Procesa la petición desde RunPod o Postman
    """
    job_input = job["input"]
    prompt = job_input.get("prompt", "A futuristic city in the style of cyberpunk")
    
    # Aseguramos que el modelo esté cargado
    load_model()
    
    # Generación de la imagen
    with torch.inference_mode():
        # Ejecutamos la inferencia con los parámetros optimizados
        output = pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=20,
            max_sequence_length=512
        )
        # Obtenemos la imagen resultante
        image = output.images[0]

    # Conversión a Base64 para el JSON de respuesta
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}

# Iniciar el worker de RunPod
runpod.serverless.start({"handler": handler})
