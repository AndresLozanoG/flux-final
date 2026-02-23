import runpod
import torch
from diffusers import FluxPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import base64
import os
from huggingface_hub import login
from io import BytesIO

# Autenticación
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Cargando modelo FLUX...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        
        # EL FIX MAESTRO: Forzamos el procesador de atención 2.0 manualmente
        # Esto reemplaza a 'set_default_attn_processor' y evita el error de 'enable_gqa'
        pipe.transformer.set_attn_processor(AttnProcessor2_0())
        
        pipe.enable_model_cpu_offload() 
        print("Modelo cargado con éxito.")

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "A futuristic city")
    load_model()
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5
        )
        image = output.images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {"image": img_str}

runpod.serverless.start({"handler": handler})
