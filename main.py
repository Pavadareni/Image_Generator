import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from fastapi import FastAPI, Query, Response
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model_id = "stabilityai/stable-diffusion-2-1"

try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
except Exception as e:
    print("Error occurred while setting up CUDA:", e)
    print("Falling back to CPU configuration.")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")

def generate_image(prompt):
    try:
        image = pipe(prompt).images[0]
    except Exception as e:
        print("Error occurred while generating the image:", e)
        return None
    return image

def clean_prompt(prompt):
    cleaned_prompt = ''.join(char.lower() for char in prompt if char.isalnum() or char.isspace())
    return cleaned_prompt

@app.get("/image/")
def get_image(prompt: str = Query(default="a photo of an astronaut riding a horse on mars")):
    cleaned_prompt = clean_prompt(prompt)
    cleaned_prompt_no_spaces = cleaned_prompt.replace(" ", "")

    folder_path = "images"
    os.makedirs(folder_path, exist_ok=True)
    image_path = os.path.join(folder_path, f"{cleaned_prompt_no_spaces}.png")

    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()
        return Response(content=image_data, media_type="image/png")
    else:
        image = generate_image(prompt)
        if image is None:
            return Response(content="Error occurred while generating the image.", status_code=500)
        
        image.save(image_path, format="PNG")

        image_io = BytesIO()
        image.save(image_io, format="PNG")
        image_io.seek(0)

        return Response(content=image_io.getvalue(), media_type="image/png")

# Author details as comments
"""
Author: HaythmKenway (akileswar)
GitHub: https://github.com/HaythmKenway
"""

# CORS settings to allow requests from localhost
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

