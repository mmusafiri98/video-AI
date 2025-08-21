import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.title("🎨 Génération d'images IA sur Streamlit")

prompt = st.text_area("📝 Décris ton image", "Un chat qui joue au piano")
width = st.slider("Largeur", 128, 512, 256)
height = st.slider("Hauteur", 128, 512, 256)

if st.button("🖼️ Générer l'image"):
    st.warning("⚠️ Génération en CPU, cela peut prendre quelques secondes")

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-tiny"
    ).to("cpu")

    image = pipe(prompt, height=height, width=width).images[0]
    st.image(image, caption="Image générée")
