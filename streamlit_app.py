import streamlit as st
from gradio_client import Client
import os
import time
import random
from PIL import Image
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Imagine AI Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
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

@st.cache_resource
def initialize_client():
    """Initialise le client Gradio"""
    try:
        client = Client("Muyumba/imagine-AI")
        return client
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation du client: {e}")
        return None

def generate_image(client, prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    """Génère une image avec le modèle Imagine AI"""
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
        
        # Le résultat peut être un tuple, une liste, ou directement un chemin
        if isinstance(result, (list, tuple)) and len(result) > 0:
            return result[0]  # Prendre le premier élément
        elif isinstance(result, str):
            return result  # Si c'est déjà un chemin de fichier
        else:
            return result  # Retourner tel quel pour d'autres formats
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération: {e}")
        return None

def display_generation_info(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed):
    """Affiche les informations de génération"""
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

def save_image_to_gallery(image_path, prompt):
    """Sauvegarde l'image dans la galerie de session"""
    if 'image_gallery' not in st.session_state:
        st.session_state.image_gallery = []
    
    # Vérifier que le chemin est valide avant d'ajouter
    if image_path and isinstance(image_path, str) and os.path.exists(image_path):
        st.session_state.image_gallery.append({
            'path': image_path,
            'prompt': prompt,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Limiter la galerie à 10 images
        if len(st.session_state.image_gallery) > 10:
            st.session_state.image_gallery = st.session_state.image_gallery[-10:]
    else:
        st.warning(f"⚠️ Impossible d'ajouter à la galerie: chemin invalide {image_path}")

def main():
    # En-tête principal
    st.markdown('<div class="main-header"><h1>🎨 Imagine AI Generator</h1><p>Créez des images époustouflantes avec l\'intelligence artificielle</p></div>', unsafe_allow_html=True)
    
    # Initialiser le client
    client = initialize_client()
    if client is None:
        st.error("❌ Impossible de se connecter au modèle. Veuillez réessayer plus tard.")
        return
    
    # Interface utilisateur
    with st.sidebar:
        st.header("🎛️ Paramètres de génération")
        
        # Paramètres de base
        st.subheader("📝 Texte")
        prompt = st.text_area(
            "Prompt (description de l'image)",
            value="A beautiful landscape with mountains and a sunset",
            height=100,
            help="Décrivez l'image que vous voulez générer"
        )
        
        negative_prompt = st.text_area(
            "Prompt négatif (ce que vous ne voulez pas)",
            value="blurry, bad quality, distorted, ugly, deformed",
            height=80,
            help="Décrivez ce que vous ne voulez PAS dans l'image"
        )
        
        # Paramètres de dimension
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            width = st.selectbox(
                "Largeur",
                options=[512, 768, 1024],
                index=0
            )
        with col2:
            height = st.selectbox(
                "Hauteur", 
                options=[512, 768, 1024],
                index=0
            )
        
        # Paramètres avancés
        st.subheader("⚙️ Paramètres avancés")
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="Plus élevé = plus fidèle au prompt"
        )
        
        num_inference_steps = st.slider(
            "Nombre d'étapes",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Plus d'étapes = meilleure qualité mais plus lent"
        )
        
        # Paramètres de seed
        randomize_seed = st.checkbox("Seed aléatoire", value=True)
        seed = st.number_input(
            "Seed (si non-aléatoire)",
            min_value=0,
            max_value=999999,
            value=0,
            disabled=randomize_seed
        )
        
        if randomize_seed:
            seed = random.randint(0, 999999)
        
        # Exemples de prompts
        st.subheader("💡 Exemples de prompts")
        example_prompts = [
            "A futuristic cityscape at night with neon lights",
            "A magical forest with glowing mushrooms and fairy lights",
            "A portrait of a cyberpunk character with colorful hair",
            "A peaceful Japanese garden with cherry blossoms",
            "An abstract digital art with vibrant colors and geometric shapes"
        ]
        
        for i, example in enumerate(example_prompts):
            if st.button(f"📝 Exemple {i+1}", key=f"example_{i}"):
                st.session_state.example_prompt = example
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🚀 Génération")
        
        # Utiliser l'exemple sélectionné si disponible
        if 'example_prompt' in st.session_state:
            prompt = st.session_state.example_prompt
            st.success(f"✅ Prompt d'exemple sélectionné: {prompt}")
            del st.session_state.example_prompt
        
        # Afficher le prompt actuel
        st.info(f"**Prompt actuel:** {prompt}")
        
        # Bouton de génération
        if st.button("🎨 Générer l'image", type="primary", use_container_width=True):
            if prompt.strip():
                start_time = time.time()
                
                # Génération de l'image
                result = generate_image(
                    client, prompt, negative_prompt, seed, randomize_seed, 
                    width, height, guidance_scale, num_inference_steps
                )
                
                if result:
                    generation_time = time.time() - start_time
                    st.success(f"✅ Image générée en {generation_time:.1f} secondes!")
                    
                    # Sauvegarder dans la session
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
                    
                    # Ajouter à la galerie
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
                # Afficher l'image générée
                image_path = st.session_state['generated_image']
                st.write(f"Debug - Chemin image: {image_path}")
                st.write(f"Debug - Type: {type(image_path)}")
                
                # Si c'est un chemin de fichier (string)
                if isinstance(image_path, str) and image_path:
                    if os.path.exists(image_path):
                        try:
                            image = Image.open(image_path)
                            st.image(image, caption="Image générée", use_column_width=True)
                            st.success("✅ Image affichée avec succès!")
                            
                            # Bouton de téléchargement
                            with open(image_path, "rb") as file:
                                st.download_button(
                                    label="💾 Télécharger l'image",
                                    data=file.read(),
                                    file_name=f"imagine_ai_{int(time.time())}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        except Exception as img_error:
                            st.error(f"❌ Erreur ouverture image: {img_error}")
                    else:
                        st.error(f"❌ Fichier non trouvé: {image_path}")
                        # Lister les fichiers du répertoire actuel pour debug
                        try:
                            files = os.listdir(".")
                            st.write(f"Fichiers disponibles: {files}")
                        except:
                            pass
                
                # Si c'est directement un objet Image PIL
                elif hasattr(image_path, 'save'):
                    st.image(image_path, caption="Image générée", use_column_width=True)
                    
                    # Convertir pour téléchargement
                    buf = io.BytesIO()
                    image_path.save(buf, format='PNG')
                    
                    st.download_button(
                        label="💾 Télécharger l'image",
                        data=buf.getvalue(),
                        file_name=f"imagine_ai_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Si c'est autre chose, essayer de traiter comme URL ou autre format
                else:
                    st.error(f"❌ Format non supporté: {type(image_path)}")
                    st.write(f"Contenu: {image_path}")
                
                # Afficher les informations de génération
                if 'generation_params' in st.session_state:
                    params = st.session_state['generation_params']
                    display_generation_info(
                        params['prompt'], params['negative_prompt'],
                        params['width'], params['height'],
                        params['guidance_scale'], params['num_inference_steps'],
                        params['seed']
                    )
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'affichage: {e}")
                st.write(f"Debug - Erreur détaillée: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.info("👈 Configurez vos paramètres et cliquez sur 'Générer l'image'")
    
    # Galerie d'images
    if 'image_gallery' in st.session_state and st.session_state.image_gallery:
        st.header("🖼️ Galerie des dernières générations")
        
        # Afficher les images en grille
        cols = st.columns(5)
        for i, img_data in enumerate(reversed(st.session_state.image_gallery[-5:])):
            with cols[i % 5]:
                try:
                    if os.path.exists(img_data['path']):
                        image = Image.open(img_data['path'])
                        st.image(image, caption=f"🕒 {img_data['timestamp']}", use_column_width=True)
                        if st.button(f"👁️ Voir", key=f"view_{i}"):
                            st.session_state['generated_image'] = img_data['path']
                            st.rerun()
                except:
                    st.write("❌ Image non disponible")
    
    # Conseils et astuces
    with st.expander("💡 Conseils pour de meilleurs résultats"):
        st.markdown("""
        ### 🎨 Conseils pour vos prompts:
        
        **✅ Prompts efficaces:**
        - Soyez spécifique et détaillé
        - Utilisez des termes artistiques (style, technique)
        - Mentionnez la qualité souhaitée (4K, high quality, detailed)
        
        **📝 Structure recommandée:**
        `[Sujet principal], [Style artistique], [Éclairage], [Couleurs], [Qualité]`
        
        **🚫 Prompts négatifs utiles:**
        - `blurry, low quality, distorted, deformed`
        - `bad anatomy, extra limbs, watermark`
        - `dark, oversaturated, cartoon` (selon vos préférences)
        
        **⚙️ Paramètres:**
        - **Guidance Scale 7-15:** Équilibre créativité/fidélité
        - **Steps 20-30:** Bon compromis qualité/vitesse
        - **Dimensions:** 512x512 pour rapidité, 1024x1024 pour qualité
        """)

if __name__ == "__main__":
    main()
