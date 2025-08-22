import streamlit as st
from gradio_client import Client

# Connexion au modèle hébergé sur Hugging Face
client = Client("Muyumba/Qwen-Qwen-Image-Edit")

st.set_page_config(page_title="Qwen Image Generator", layout="centered")

st.title("🎨 Qwen Image Generator")
st.write("Télécharge une image et décris le style que tu veux. Le modèle génère automatiquement une nouvelle image.")

# Champ texte pour la description
instruction = st.text_area("📝 Décris le type d’image que tu veux générer :", 
                           placeholder="Exemple : transforme l’image en style dessin animé...")

# Téléversement d’image
uploaded_file = st.file_uploader("📤 Télécharge une image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

# Si l'utilisateur a donné une image ET une description
if uploaded_file is not None and instruction.strip():
    st.image(uploaded_file, caption="📸 Image originale", use_column_width=True)

    with st.spinner("⏳ Génération en cours..."):
        try:
            # Appel au modèle Hugging Face
            result = client.predict(
                uploaded_file,   # image téléversée
                instruction,     # description donnée par l’utilisateur
                api_name="/predict"
            )

            # Affichage du résultat
            if isinstance(result, str):  # si le modèle retourne une URL ou un chemin
                st.image(result, caption="🖼️ Image générée", use_column_width=True)
            elif isinstance(result, list) and len(result) > 0:
                st.image(result[0], caption="🖼️ Image générée", use_column_width=True)
            else:
                st.error("❌ Résultat inattendu du modèle :")
                st.write(result)

        except Exception as e:
            st.error(f"Erreur lors de la génération : {e}")

