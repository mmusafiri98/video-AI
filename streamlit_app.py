import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Charger le modèle une seule fois (mise en cache Streamlit)
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

st.title("🎬 Générateur de Vidéo avec Diffusers")

# Upload image de départ
image_file = st.file_uploader("📤 Uploade une image de départ (PNG/JPG)", type=["png", "jpg", "jpeg"])

if image_file:
    from PIL import Image
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Image de départ", use_container_width=True)

    # Bouton de génération
    if st.button("🚀 Générer la vidéo"):
        pipe = load_pipeline()

        with st.spinner("Génération en cours... ⏳"):
            video_frames = pipe(image, num_frames=16).frames

        # Sauvegarder la vidéo temporairement
        import imageio
        output_path = "output.mp4"
        imageio.mimsave(output_path, video_frames, fps=8)

        st.video(output_path)
        st.success("✅ Vidéo générée avec succès !")



