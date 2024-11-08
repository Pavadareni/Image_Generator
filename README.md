# Stable Diffusion Image Generator with Streamlit

This project is a Streamlit app that uses the **Stable Diffusion** model to generate images from text prompts. The app allows users to input a custom prompt, generate an image based on that prompt, and download the generated image. Generated images are cached for efficiency, so repeated prompts load faster.

## Features
- **Text-to-Image Generation**: Uses Stable Diffusion to generate images based on a text prompt.
- **Caching**: Images are cached in the `images` folder. Repeated prompts will load the cached images instead of generating them again.
- **Download**: Users can download generated images in PNG format.
- **CPU/GPU Support**: Automatically switches between CPU and GPU depending on availability.

## Installation

To set up and run the project locally:

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/Pavadareni/Image_Generator.git
   cd Image_Generator
2 **Install Dependencies**
Ensure Python is installed on your system. Install the required dependencies:
 ```bash
pip install -r requirements.txt

```
3 **Running the App Locally**
To run the Streamlit app locally:
Run the Streamlit App: Open a terminal, navigate to the project directory, and run:
 ```bash
streamlit run main.py
```
