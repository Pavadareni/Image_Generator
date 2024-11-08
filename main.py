import os
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from io import BytesIO
import hashlib

# Set up model ID and folder for storing images
model_id = "stabilityai/stable-diffusion-2-1"
folder_path = "images"
os.makedirs(folder_path, exist_ok=True)

# Load Stable Diffusion pipeline with a fallback to CPU if CUDA is unavailable
@st.cache_resource  # Cache model to avoid reloading it for every request
def load_pipeline():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
    except Exception as e:
        st.warning(f"CUDA is not available. Falling back to CPU. Error: {e}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cpu")
    return pipe

pipe = load_pipeline()  # Load pipeline once and cache it

# Function to generate an image based on the prompt
def generate_image(prompt):
    try:
        image = pipe(prompt).images[0]
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None
    return image

# Function to create a hash for each prompt for unique filenames
def generate_image_filename(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()

# Streamlit Interface
st.title("Stable Diffusion Image Generator")
prompt = st.text_input(
    "Enter a prompt:", value="A photo of an astronaut riding a horse on Mars"
)

# Cache the result of image generation based on the prompt (so it doesn't regenerate on each run)
if st.button("Generate Image"):
    # Clean and hash the prompt to use as a unique filename
    cleaned_prompt = prompt.lower().strip()
    image_filename = generate_image_filename(cleaned_prompt)
    image_path = os.path.join(folder_path, f"{image_filename}.png")

    # Check if image already exists in cache
    if os.path.exists(image_path):
        st.write("Image loaded from cache.")
        image = Image.open(image_path)
    else:
        # Generate a new image and save it
        image = generate_image(prompt)
        if image:
            image.save(image_path, format="PNG")
        else:
            st.error("Failed to generate image.")
            image = None

    # Display the generated image if it exists
    if image:
        st.image(
            image,
            caption=f"Generated image for prompt: '{prompt}'",
            use_column_width=True,
        )

        # Provide a download button for the image
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        st.download_button(
            label="Download Image",
            data=image_bytes.getvalue(),
            file_name=f"{image_filename}.png",
            mime="image/png",
        )
