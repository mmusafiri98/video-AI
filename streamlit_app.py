import os
import shutil
import tempfile
import re
from datetime import datetime
import streamlit as st

# -------------------------------
# Dépendances et vérification
# -------------------------------
def check_dependencies():
    deps_status = {}
    modules = {}
    try:
        import torch
        deps_status['torch'] = True
        modules['torch'] = torch
        torch_version = torch.__version__
    except ImportError:
        deps_status['torch'] = False
        modules['torch'] = None
        torch_version = "Non installé"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        import transformers
        deps_status['transformers'] = True
        modules['transformers'] = {
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'AutoTokenizer': AutoTokenizer,
            'StoppingCriteria': StoppingCriteria,
            'StoppingCriteriaList': StoppingCriteriaList,
            'transformers': transformers
        }
        transformers_version = transformers.__version__
    except ImportError:
        deps_status['transformers'] = False
        modules['transformers'] = None
        transformers_version = "Non installé"
    return deps_status, torch_version, transformers_version, modules

DEPS_STATUS, TORCH_VERSION, TRANSFORMERS_VERSION, MODULES = check_dependencies()
TORCH_AVAILABLE = DEPS_STATUS['torch']
TRANSFORMERS_AVAILABLE = DEPS_STATUS['transformers']

# -------------------------------
# Critères de fin de génération
# -------------------------------
if TRANSFORMERS_AVAILABLE and MODULES['transformers']:
    StoppingCriteria = MODULES['transformers']['StoppingCriteria']
    class SmartSentenceStopping(StoppingCriteria):
        def __init__(self, tokenizer, min_length=50):
            self.tokenizer = tokenizer
            self.min_length = min_length
            
        def __call__(self, input_ids, scores, **kwargs):
            # Ne s'arrête que si le modèle génère naturellement un token de fin
            # ou si on atteint la limite absolue de longueur
            return False  # Laisse le modèle décider quand s'arrêter
else:
    class SmartSentenceStopping:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return False

# -------------------------------
# Gestion du cache
# -------------------------------
class CacheManager:
    def __init__(self):
        self.cache_dir = None
        self.setup_cache()
    def setup_cache(self):
        if TRANSFORMERS_AVAILABLE:
            self.cache_dir = tempfile.mkdtemp(prefix="colegium_cache_")
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HF_DATASETS_CACHE"] = self.cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cleanup_locks()
    def cleanup_locks(self):
        if not self.cache_dir: return
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith('.lock'):
                    try: os.remove(os.path.join(root, file))
                    except: pass
    def cleanup_cache(self):
        if self.cache_dir and os.path.exists(self.cache_dir):
            try: shutil.rmtree(self.cache_dir)
            except: pass

cache_manager = CacheManager()

# -------------------------------
# Traitement de texte
# -------------------------------
class TextProcessor:
    @staticmethod
    def detect_code_language(response):
        """Détecte le langage de programmation dans la réponse"""
        response_lower = response.lower()
        
        # Détection par mots-clés spécifiques
        if any(keyword in response_lower for keyword in ['<html', '<div', '<body', '<head', '<!doctype']):
            return 'html'
        elif any(keyword in response_lower for keyword in ['def ', 'import ', 'print(', 'if __name__', 'class ']):
            return 'python'
        elif any(keyword in response_lower for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log', '=>']):
            return 'javascript'
        elif any(keyword in response_lower for keyword in ['public class', 'public static void', 'system.out']):
            return 'java'
        elif any(keyword in response_lower for keyword in ['#include', 'int main', 'cout', 'cin', 'std::']):
            return 'cpp'
        elif any(keyword in response_lower for keyword in ['fn ', 'let mut', 'println!', 'match ']):
            return 'rust'
        elif any(keyword in response_lower for keyword in ['func ', 'package main', 'fmt.print']):
            return 'go'
        elif any(keyword in response_lower for keyword in ['<?php', 'echo ', '$_']):
            return 'php'
        elif any(keyword in response_lower for keyword in ['select ', 'from ', 'where ', 'insert ', 'update ']):
            return 'sql'
        elif any(keyword in response_lower for keyword in ['background:', 'color:', 'font-size:', '.class', '#id']):
            return 'css'
        elif any(keyword in response_lower for keyword in ['class Program', 'Console.Write', 'using System']):
            return 'csharp'
        elif any(keyword in response_lower for keyword in ['puts ', 'def ', 'end', '@']):
            return 'ruby'
        elif any(keyword in response_lower for keyword in ['echo ', 'bash', '#!/bin/']):
            return 'bash'
        else:
            return 'text'

    @staticmethod
    def format_code_by_language(response, language):
        """Formate le code selon le langage détecté"""
        if language == 'html':
            # Formatage HTML
            response = re.sub(r'(<[^/>][^>]*>)', r'\1\n', response)  # Saut après balises ouvrantes
            response = re.sub(r'(</[^>]+>)', r'\1\n', response)      # Saut après balises fermantes
            response = re.sub(r'(<[^/>][^>]*/>)', r'\1\n', response) # Saut après balises auto-fermantes
            
        elif language in ['css']:
            # Formatage CSS
            response = re.sub(r'(\{)', r' \1\n', response)      # Saut après {
            response = re.sub(r'(\})', r'\n\1\n', response)     # Saut avant et après }
            response = re.sub(r'(;)', r'\1\n  ', response)      # Saut et indentation après ;
            response = re.sub(r'(:)', r'\1 ', response)         # Espace après :
            
        elif language in ['javascript', 'java', 'cpp', 'csharp', 'php']:
            # Formatage langages avec accolades
            response = re.sub(r'(\{)', r' \1\n  ', response)    # Saut et indentation après {
            response = re.sub(r'(\})', r'\n\1\n', response)     # Saut avant et après }
            response = re.sub(r'(;)', r'\1\n  ', response)      # Saut et indentation après ;
            response = re.sub(r'(\n  ){2,}', r'\n  ', response) # Évite double indentation
            
        elif language == 'python':
            # Formatage Python
            response = re.sub(r'(:)', r'\1\n    ', response)    # Saut et indentation après :
            response = re.sub(r'(\n    ){2,}', r'\n    ', response) # Évite double indentation
            
        elif language in ['sql']:
            # Formatage SQL
            response = re.sub(r'(\bSELECT\b)', r'\n\1', response, flags=re.IGNORECASE)
            response = re.sub(r'(\bFROM\b)', r'\n\1', response, flags=re.IGNORECASE)
            response = re.sub(r'(\bWHERE\b)', r'\n\1', response, flags=re.IGNORECASE)
            response = re.sub(r'(;)', r'\1\n', response)
            
        # Nettoyage final
        response = re.sub(r'\n\s*\n', '\n', response)          # Supprime lignes vides multiples
        response = response.strip()
        
        return response

    @staticmethod
    def is_code_content(response):
        """Vérifie si le contenu contient du code"""
        code_indicators = [
            # HTML/XML
            '<html', '<div', '<body', '<head', '<script', '<style',
            # Python
            'def ', 'import ', 'class ', 'if __name__',
            # JavaScript
            'function ', 'const ', 'let ', 'var ', '=>',
            # Java/C#
            'public class', 'public static', 'System.out',
            # CSS
            'background:', 'color:', 'font-size:', '.class', '#id',
            # SQL
            'SELECT ', 'FROM ', 'WHERE ', 'INSERT ',
            # Autres
            'cout', 'println!', '<?php', 'echo ', 'bash'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in code_indicators)

    @staticmethod
    def complete_sentences(text):
        if not text.strip(): return text
        endings = ['.', '!', '?', '...']
        for i in range(len(text) - 1, -1, -1):
            if text[i] in endings:
                return text[:i+1]
        if text[-1] in [',', ';', ':']:
            return text[:-1] + '.'
        return text + '.'
    
    @staticmethod
    def clean_response(response):
        response = re.sub(r'^(Assistant|AI|Bot):\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'[ ]{2,}', ' ', response)
        return response.strip()

text_processor = TextProcessor()

# -------------------------------
# Chargement du modèle
# -------------------------------
@st.cache_resource
def load_model_enhanced():
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
        return None, None, None, "Dependencies missing"
    try:
        AutoModelForCausalLM = MODULES['transformers']['AutoModelForCausalLM']
        AutoTokenizer = MODULES['transformers']['AutoTokenizer']
        torch = MODULES['torch']

        model_name = "Muyumba/colegium-ai"

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_manager.cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_manager.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available(): model.to(device)
        return tokenizer, model, device, "success"
    except Exception as e:
        return None, None, None, f"error: {str(e)}"

# -------------------------------
# Fallback simple
# -------------------------------
class FallbackResponder:
    responses = {
        "greeting": ["Bonjour !", "Salut !", "Bonsoir !"],
        "question": ["C'est une question intéressante.", "Votre question mérite réflexion."],
        "help": ["Je suis là pour vous aider.", "Pouvez-vous préciser ?"],
        "default": ["Merci pour votre message.", "Je prends note."]
    }
    def get_response(self, user_input):
        user_lower = user_input.lower()
        if any(w in user_lower for w in ['bonjour', 'salut', 'hello']): category = "greeting"
        elif any(w in user_lower for w in ['?', 'comment', 'pourquoi']): category = "question"
        elif any(w in user_lower for w in ['aide', 'help']): category = "help"
        else: category = "default"
        import random
        resp = random.choice(self.responses[category])
        if len(user_input.split()) > 20:
            resp += " Votre message est détaillé."
        return resp

fallback_responder = FallbackResponder()

# -------------------------------
# Interface Streamlit
# -------------------------------
st.set_page_config(page_title="Colegium-AI", page_icon="🤖", layout="centered")
st.title("🤖 Colegium-AI - Assistant Conversationnel")

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    tokenizer, model, device, load_status = load_model_enhanced()
else:
    tokenizer, model, device, load_status = None, None, None, "Dependencies missing"

if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_stats" not in st.session_state:
    st.session_state.conversation_stats = {"total_messages": 0, "total_words": 0, "session_start": datetime.now()}

# Affichage status
if "success" in load_status:
    st.success(f"✅ Modèle chargé: {device}")
elif load_status == "Dependencies missing":
    st.error("❌ PyTorch et Transformers requis")
else:
    st.error(f"❌ Erreur: {load_status}")

# CORRECTION: D'abord afficher l'historique existant
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# Chat input
if prompt := st.chat_input("Écrivez votre message ici..."):
    # Ajouter le message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_stats["total_messages"] += 1
    st.session_state.conversation_stats["total_words"] += len(prompt.split())
    
    # Afficher le message utilisateur
    with st.chat_message("user"):
        st.write(prompt)

    # Generazione risposta avec spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤖 Le modèle réfléchit..."):
            if model and tokenizer:
                try:
                    from transformers import StoppingCriteriaList
                    conversation = "\n".join(
                        f"{'Utilisateur' if m['role']=='user' else 'Assistant'}: {m['content']}" 
                        for m in st.session_state.messages[-6:]
                    ) + "\nAssistant:"
                    # Pas de stopping criteria artificiel - laisse le modèle décider
                    inputs = tokenizer.encode(conversation, return_tensors="pt", max_length=1024, truncation=True).to(device)
                    output_ids = model.generate(
                        inputs, 
                        max_length=2048,     # Limite la longueur totale (contexte + génération)
                        do_sample=True,      # Active le sampling pour plus de variété
                        temperature=0.7,     # Contrôle la créativité
                        top_p=0.9,          # Nucleus sampling
                        repetition_penalty=1.1,  # Évite les répétitions
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    response = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
                    response = text_processor.clean_response(response)
                    
                    # Détection et formatage du code
                    if text_processor.is_code_content(response):
                        language = text_processor.detect_code_language(response)
                        response = text_processor.format_code_by_language(response, language)
                    else:
                        response = text_processor.complete_sentences(response)
                except Exception as e:
                    response = fallback_responder.get_response(prompt)
            else:
                response = fallback_responder.get_response(prompt)
        
        # Afficher la réponse avec le bon formatage
        if text_processor.is_code_content(response):
            language = text_processor.detect_code_language(response)
            # Affiche le code avec coloration syntaxique appropriée
            message_placeholder.code(response, language=language if language != 'text' else None)
        else:
            # Affichage normal pour le texte
            message_placeholder.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Statistiques
st.divider()
stats = st.session_state.conversation_stats
st.write(f"Messages: {stats['total_messages']} | Mots: {stats['total_words']} | Début session: {stats['session_start']}")

# Export conversation
if st.button("Exporter la conversation"):
    txt_content = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages)
    st.download_button("Télécharger .txt", txt_content, "conversation.txt", "text/plain")

# Nouvelle conversation
if st.button("Nouvelle conversation"):
    st.session_state.messages = []
    st.session_state.conversation_stats = {"total_messages": 0, "total_words": 0, "session_start": datetime.now()}
    st.rerun()  # Changé de st.experimental_rerun() à st.rerun()
