# src/streamlit_app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

st.set_page_config(page_title="Mini Video Generator", layout="centered")

st.title("🎬 Mini Video Generator")
st.write("Génère des vidéos courtes à partir de prompts textuels (100% local et léger).")

# Prompt utilisateur
prompt = st.text_input(
    "📝 Décris la vidéo :",
    placeholder="Ex: Un chat qui danse dans une ville futuriste"
)

num_frames = st.slider("Nombre de frames", 3, 10, 5)
duration = st.slider("Durée d'une frame (ms)", 200, 1000, 500)

if st.button("🚀 Générer"):
    if not prompt.strip():
        st.warning("Veuillez entrer une description !")
    else:
        with st.spinner("Génération en cours..."):
            # Charger un modèle SD léger
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                cache_dir="/app/.cache"  # ✅ évite /root/.cache interdit
            ).to(device)

            frames = []
            for i in range(num_frames):
                image = pipe(
                    prompt,
                    num_inference_steps=15,
                    guidance_scale=7.5
                ).images[0]
                frames.append(image)

            # Sauvegarder en GIF
            gif_path = Path("generated_video.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )

            st.success("✅ GIF généré !")
            st.image(str(gif_path))

            # --- Télécharger le GIF en bytes ---
            with open(gif_path, "rb") as f:
                gif_bytes = f.read()

            st.download_button(
                label="⬇️ Télécharger le GIF",
                data=gif_bytes,
                file_name="generated_video.gif",  # ✅ string
                mime="image/gif"
            )





