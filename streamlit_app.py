import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("🤖 Text-Generation CPU Demo")

model_name = "distilgpt2"  # Modèle léger
device = "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Écris ton texte ici :", "Bonjour, comment ça va ?")

max_length = st.slider("Longueur max de génération", 20, 200, 50)

if st.button("Générer"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with st.spinner("Génération en cours..."):
        output = model.generate(inputs["input_ids"], max_length=max_length)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.text_area("Texte généré :", text, height=200)

