import runpod
import torch
from diffusers import FluxPipeline
import base64
import os
from huggingface_hub import login
from io import BytesIO

# Autenticación segura
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Variable global para el modelo
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Cargando modelo con optimización de memoria...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        ) # Quitamos el .to("cuda") de aquí
        
        # FIX DE MEMORIA: Mueve partes del modelo entre CPU y GPU automáticamente
        pipe.enable_model_cpu_offload() 
        
        # FIX DE ATENCIÓN: (El que ya teníamos)
        pipe.transformer.set_default_attn_processor()
        print("Modelo cargado con éxito usando CPU Offloading.")

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "A futuristic city in the style of cyberpunk")
    
    load_model()
    
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=20,
            max_sequence_length=512
        )
        image = output.images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str}

runpod.serverless.start({"handler": handler})
