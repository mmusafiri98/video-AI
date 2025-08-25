import streamlit as st
from gradio_client import Client
import os
import time
import random
from PIL import Image
import io
import json

# --- Configuration ---
GALLERY_DIR = "gallery"
GALLERY_JSON = os.path.join(GALLERY_DIR, "gallery.json")

st.set_page_config(
    page_title="Imagine AI Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .generation-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions persistance ---
def load_gallery():
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)
    if os.path.exists(GALLERY_JSON):
        with open(GALLERY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_gallery(gallery):
    with open(GALLERY_JSON, "w", encoding="utf-8") as f:
        json.dump(gallery, f, ensure_ascii=False, indent=2)

def save_image_to_gallery(image_path, prompt):
    """Sauvegarde l'image et ses infos dans la galerie persistante"""
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)

    if image_path and os.path.exists(image_path):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        new_path = os.path.join(GALLERY_DIR, f"gen_{timestamp}.png")

        try:
            img = Image.open(image_path)
            img.save(new_path, "PNG")

            gallery = load_gallery()
            gallery.append({
                "path": new_path,
                "prompt": prompt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            # Garder max 50 images
            gallery = gallery[-50:]
            save_gallery(gallery)

        except Exception as e:
            st.error(f"Erreur sauvegarde image: {e}")

# --- Client Gradio ---
@st.cache_resource
def initialize_client():
    try:
        client = Client("Muyumba/imagine-AI")
        return client
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation du client: {e}")
        return None

def generate_image(client, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    try:
        with st.spinner("🎨 Génération de votre image en cours..."):
            result = client.predict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                randomize_seed=randomize_seed,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                api_name="/infer"
            )
        if isinstance(result, (list, tuple)) and len(result) > 0:
            return result[0]
        elif isinstance(result, str):
            return result
        else:
            return result
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération: {e}")
        return None

def display_generation_info(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed):
    st.markdown('<div class="generation-info">', unsafe_allow_html=True)
    st.markdown("**📋 Paramètres de génération:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Prompt:** {prompt}")
        st.write(f"**Prompt négatif:** {negative_prompt}")
        st.write(f"**Dimensions:** {width}x{height}")
    with col2:
        st.write(f"**Guidance Scale:** {guidance_scale}")
        st.write(f"**Steps:** {num_inference_steps}")
        st.write(f"**Seed:** {seed}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Application principale ---
def main():
    st.markdown('<div class="main-header"><h1>🎨 Imagine AI Generator</h1><p>Créez des images époustouflantes avec l\'intelligence artificielle</p></div>', unsafe_allow_html=True)

    client = initialize_client()
    if client is None:
        st.error("❌ Impossible de se connecter au modèle. Veuillez réessayer plus tard.")
        return

    with st.sidebar:
        st.header("🎛️ Paramètres de génération")
        st.subheader("📝 Texte")
        prompt = st.text_area("Prompt", value="A beautiful landscape with mountains and a sunset", height=100)
        negative_prompt = st.text_area("Prompt négatif", value="blurry, bad quality, distorted, ugly, deformed", height=80)

        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox("Largeur", options=[512, 768, 1024], index=0)
        with col2:
            height = st.selectbox("Hauteur", options=[512, 768, 1024], index=0)

        st.subheader("⚙️ Paramètres avancés")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        num_inference_steps = st.slider("Nombre d'étapes", 10, 50, 20, 5)

        randomize_seed = st.checkbox("Seed aléatoire", value=True)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=0, disabled=randomize_seed)
        if randomize_seed:
            seed = random.randint(0, 999999)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🚀 Génération")
        if st.button("🎨 Générer l'image", type="primary", use_container_width=True):
            if prompt.strip():
                start_time = time.time()
                result = generate_image(client, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)
                if result:
                    generation_time = time.time() - start_time
                    st.success(f"✅ Image générée en {generation_time:.1f} secondes!")
                    st.session_state['generated_image'] = result
                    st.session_state['generation_params'] = {
                        'prompt': prompt,
                        'negative_prompt': negative_prompt,
                        'width': width,
                        'height': height,
                        'guidance_scale': guidance_scale,
                        'num_inference_steps': num_inference_steps,
                        'seed': seed
                    }
                    save_image_to_gallery(result, prompt)
                    st.rerun()
                else:
                    st.error("❌ Échec de la génération")
            else:
                st.warning("⚠️ Veuillez entrer un prompt")

    with col2:
        st.header("🖼️ Résultat")
        if 'generated_image' in st.session_state:
            try:
                image_path = st.session_state['generated_image']
                if isinstance(image_path, str) and os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption="Image générée", use_column_width=True)
                    with open(image_path, "rb") as file:
                        st.download_button("💾 Télécharger", file.read(), file_name=f"imagine_ai_{int(time.time())}.png", mime="image/png", use_container_width=True)
                elif hasattr(image_path, 'save'):
                    st.image(image_path, caption="Image générée", use_column_width=True)
                    buf = io.BytesIO()
                    image_path.save(buf, format='PNG')
                    st.download_button("💾 Télécharger", buf.getvalue(), file_name=f"imagine_ai_{int(time.time())}.png", mime="image/png", use_container_width=True)
                if 'generation_params' in st.session_state:
                    p = st.session_state['generation_params']
                    display_generation_info(p['prompt'], p['negative_prompt'], p['width'], p['height'], p['guidance_scale'], p['num_inference_steps'], p['seed'])
            except Exception as e:
                st.error(f"❌ Erreur affichage: {e}")
        else:
            st.info("👈 Configurez vos paramètres et cliquez sur 'Générer l'image'")

    # Galerie persistante
    gallery = load_gallery()
    if gallery:
        st.header("🖼️ Galerie des dernières générations")
        cols = st.columns(5)
        for i, img_data in enumerate(reversed(gallery[-10:])):
            with cols[i % 5]:
                try:
                    if os.path.exists(img_data['path']):
                        image = Image.open(img_data['path'])
                        st.image(image, caption=f"🕒 {img_data['timestamp']}", use_column_width=True)
                        if st.button(f"👁️ Voir {i}", key=f"view_{i}"):
                            st.session_state['generated_image'] = img_data['path']
                            st.rerun()
                except:
                    st.write("❌ Image non disponible")

if __name__ == "__main__":
    main()

            
