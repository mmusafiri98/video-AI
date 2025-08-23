# Alternative pour tester le client directement
import streamlit as st
from gradio_client import Client
import os

st.title("🔧 Test du client Gradio")

# Test direct du client
if st.button("Tester la connexion"):
    try:
        client = Client("Muyumba/imagine-AI")
        st.success("✅ Connexion réussie!")
        
        # Test avec paramètres simples
        with st.spinner("Test de génération..."):
            result = client.predict(
                prompt="a cat",
                negative_prompt="blurry",
                seed=42,
                randomize_seed=False,
                width=512,
                height=512,
                guidance_scale=7.5,
                num_inference_steps=10,  # Réduit pour test rapide
                api_name="/infer"
            )
        
        st.write(f"**Type du résultat:** {type(result)}")
        st.write(f"**Contenu:** {result}")
        
        # Essayer différentes méthodes d'accès
        if isinstance(result, (list, tuple)):
            st.write(f"**Longueur:** {len(result)}")
            for i, item in enumerate(result):
                st.write(f"**Item {i}:** {type(item)} - {item}")
                
                # Si c'est un chemin de fichier
                if isinstance(item, str) and os.path.exists(item):
                    from PIL import Image
                    image = Image.open(item)
                    st.image(image, caption=f"Résultat {i}")
        
        elif isinstance(result, str):
            if os.path.exists(result):
                from PIL import Image
                image = Image.open(result)
                st.image(image, caption="Image générée")
            else:
                st.error(f"Fichier non trouvé: {result}")
        
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        import traceback
        st.code(traceback.format_exc())
