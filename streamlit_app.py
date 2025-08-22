import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import imageio
import tempfile
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Video Generator CPU", page_icon="🎬", layout="centered")

st.title("🎬 Générateur Vidéo CPU-friendly")

# ---------- Charger le modèle (open-source) ----------
@st.cache_resource
def load_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe.to("cpu")  # CPU uniquement
    return pipe

pipe = load_model()

# ---------- Upload d'image ----------
uploaded_file = st.file_uploader("📤 Uploade une image de départ", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image de départ", use_container_width=True)

    prompt = st.text_input("📝 Décris la transformation à appliquer", "Un paysage futuriste cyberpunk")
    num_frames = st.slider("⏱ Nombre de frames pour la vidéo", 4, 12, 6)

    if st.button("🚀 Générer la vidéo"):
        with st.spinner("Génération en cours... ⏳ (CPU peut être lent)"):
            frames = []
            for i in range(num_frames):
                # Chaque frame peut être légèrement différente si tu veux un effet animé
                frame = pipe(prompt=prompt, image=image, strength=0.6, guidance_scale=7.5).images[0]
                frames.append(frame)

            # Sauvegarde en vidéo
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "output.mp4")
            imageio.mimsave(output_path, frames, fps=4)  # fps bas pour CPU

        st.success("✅ Vidéo générée !")
        st.video(output_path)

