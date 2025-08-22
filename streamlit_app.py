import streamlit as st
from gradio_client import Client

# Charger le client Hugging Face Spaces
client = Client("Muyumba/Qwen-Qwen-Image-Edit")

st.set_page_config(page_title="Qwen Image Edit", layout="centered")

st.title("🖼️ Qwen Image Edit - Demo")
st.write("Application Streamlit connectée au modèle hébergé sur Hugging Face Spaces.")

# Zone de téléchargement d'image
uploaded_file = st.file_uploader("📤 Télécharge une image", type=["png", "jpg", "jpeg"])

# Champ texte pour instruction
instruction = st.text_input("✏️ Décris la modification que tu veux appliquer à l'image :")

if uploaded_file is not None and instruction:
    st.image(uploaded_file, caption="Image originale", use_column_width=True)

    # Bouton de soumission
    if st.button("🚀 Modifier l'image"):
        with st.spinner("⏳ En cours de traitement..."):
            # Envoyer l'image et l'instruction au modèle
            result = client.predict(
                uploaded_file,      # l’image uploadée
                instruction,        # le texte d'édition
                api_name="/predict" # endpoint de Spaces (souvent "/predict")
            )

        # Afficher le résultat
        if isinstance(result, str):  # si c’est un chemin ou une URL
            st.image(result, caption="Image modifiée", use_column_width=True)
        else:
            st.write("Résultat brut :", result)
