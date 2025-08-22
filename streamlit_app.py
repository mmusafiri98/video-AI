import gradio as gr
import requests
from PIL import Image
from io import BytesIO

HF_API_TOKEN = "hf_FqcdfTQhZRqEnLyDSdSZNUHIGQGkcfWIomE"  # Remplace par ton token Hugging Face

def generate_image(prompt):
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}
    
    response = requests.post(url, headers=headers, json=payload)
    image_bytes = response.content
    image = Image.open(BytesIO(image_bytes))
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Générateur d'images Stable Diffusion",
    description="Crée une image à partir d'un prompt via Hugging Face API"
)

iface.launch()
