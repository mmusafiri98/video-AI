import streamlit as st
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import imageio

# Charger un modèle image → image (plus léger que vidéo complète)
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")  # CPU uniquement
    return pipe

st.title("🎬 Générateur Vidéo (CPU-friendly)")

uploaded_file = st.file_uploader("📤 Uploade une image de départ", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image de départ", use_container_width=True)

    prompt = st.text_input("📝 Décris la transformation à appliquer", "Un paysage futuriste cyberpunk")

    if st.button("🚀 Générer la vidéo (courte animation)"):
        pipe = load_pipeline()

        frames = []
        with st.spinner("Génération en cours... ⏳ (ça peut prendre ~1-2 minutes)"):
            for i in range(8):  # 8 frames max pour rester léger
                frame = pipe(prompt=prompt, image=image, strength=0.6, guidance_scale=7.5).images[0]
                frames.append(frame)

        output_path = "output.mp4"
        imageio.mimsave(output_path, frames, fps=4)  # fps bas pour compresser

        st.video(output_path)
        st.success("✅ Vidéo générée avec succès !")
