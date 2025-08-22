import streamlit as st
from gradio_client import Client

# Connexion au Space Hugging Face
client = Client("ovi054/Qwen-Image-LORA")

st.set_page_config(page_title="Qwen Image Generator", layout="centered")

st.title("🎨 Qwen Image Generator (LORA)")
st.write("Génère des images à partir de texte avec le modèle **Qwen-Image-LORA** hébergé sur Hugging Face.")

# Prompt utilisateur
prompt = st.text_area("📝 Décris l'image que tu veux générer :", 
                      placeholder="Exemple : Un chat qui joue de la guitare dans l'espace...")

# Paramètres de génération
col1, col2 = st.columns(2)
with col1:
    width = st.selectbox("📏 Largeur", [512, 768, 1024], index=2)
    guidance_scale = st.slider("🎯 Guidance Scale", 1.0, 20.0, 4.0)
with col2:
    height = st.selectbox("📐 Hauteur", [512, 768, 1024], index=2)
    steps = st.slider("🔄 Steps d'inférence", 10, 50, 28)

# Seed (aléatoire ou fixe)
randomize_seed = st.checkbox("🎲 Random Seed", value=True)
seed = -1 if randomize_seed else st.number_input("Seed fixe :", value=42, step=1)

# Bouton pour lancer la génération
if st.button("🚀 Générer l'image") and prompt.strip():
    with st.spinner("⏳ Génération en cours..."):
        try:
            result = client.predict(
                prompt=prompt,
                seed=seed,
                randomize_seed=randomize_seed,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                lora_id=None,
                lora_scale=1,
                api_name="/infer"
            )

            # Résultat = URL ou chemin vers l'image
            if isinstance(result, str):
                st.image(result, caption="🖼️ Image générée", use_column_width=True)
            elif isinstance(result, list) and len(result) > 0:
                st.image(result[0], caption="🖼️ Image générée", use_column_width=True)
            else:
                st.error("❌ Résultat inattendu du modèle")
                st.write(result)

        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")




