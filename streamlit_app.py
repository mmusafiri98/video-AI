import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import imageio
import tempfile

st.set_page_config(page_title="Mini Vidéo IA", page_icon="🎬", layout="centered")
st.title("🎬 Mini génération vidéo (images animées)")

prompt = st.text_area("📝 Décris ta scène", "Un coucher de soleil au bord de la mer")
frames_count = st.slider("🖼️ Nombre d'images", 4, 12, 6)

if st.button("🚀 Générer (CPU léger)"):
    st.warning("⚠️ Cela peut prendre un peu de temps (CPU)")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-4"
    ).to("cpu")

    frames = []
    for i in range(frames_count):
        image = pipe(prompt).images[0]
        frames.append(image)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    imageio.mimsave(tmp_file.name, frames, fps=4)

    st.video(tmp_file.name)
    st.success("✅ Vidéo (animation) générée avec succès !")
