import streamlit as st
from gradio_client import Client
import requests
import os

# Charger le client Hugging Face Spaces
client = Client("Muyumba/Qwen-Qwen-Image-Edit")

st.set_page_config(page_title="Qwen Image Generator", layout="centered")

st.title("🎨 Qwen Image Generator")
st.write("Décris l'image que tu veux, et le modèle va la générer.")

# Entrée texte de l'utilisateur
prompt = st.text_area("✏️ Décris ton image :", placeholder="Exemple : un chat jouant de la guitare dans l’espace")

if st.button("🚀 Générer l'image") and prompt:
    with st.spinner("⏳ Génération en cours..."):
        # Envoi du texte au modèle Hugging Face
        result = client.predict(
            prompt,             # description
            api_name="/predict" # endpoint Spaces (souvent /predict)
        )

    # Affichage de l'image
    if isinstance(result, str):
        st.image(result, caption="🖼️ Image générée", use_column_width=True)

        # Télécharger l'image
        try:
            response = requests.get(result)
            filename = "generated_image.png"
            with open(filename, "wb") as f:
                f.write(response.content)

            with open(filename, "rb") as f:
                st.download_button(
                    label="📥 Télécharger l'image",
                    data=f,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            os.remove(filename)
        except Exception as e:
            st.error(f"Erreur téléchargement : {e}")
    else:
        st.write("Résultat brut :", result)

