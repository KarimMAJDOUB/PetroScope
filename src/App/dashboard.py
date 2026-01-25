import streamlit as st
import os
import sys

# On ajoute le dossier courant au chemin pour trouver auth_manager
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from auth_manager import create_user, check_login

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="PetroScope AI", page_icon="🛢️", layout="wide")

# CSS pour faire joli
st.markdown("""
<style>
    .main-header {font-size: 30px; font-weight: bold; color: #FF4B4B;}
    .success-msg {color: green; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- GESTION DE SESSION (Mémoire) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# --- FONCTION 1 : LA PAGE DE CONNEXION ---
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🛢️ PetroScope Login")
        st.write("Bienvenue sur la plateforme de prédiction pétrolière.")
        
        tab1, tab2 = st.tabs(["🔐 Se Connecter", "📝 S'inscrire"])
        
        # --- ONGLET CONNEXION ---
        with tab1:
            with st.form("login_form"):
                user = st.text_input("Identifiant")
                pwd = st.text_input("Mot de passe", type="password")
                submit = st.form_submit_button("Entrer", use_container_width=True)
                
                if submit:
                    if check_login(user, pwd):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = user
                        st.rerun()
                    else:
                        st.error("Identifiant ou mot de passe incorrect.")

        # --- ONGLET INSCRIPTION ---
        with tab2:
            with st.form("register_form"):
                st.write("Créer un nouveau compte")
                new_user = st.text_input("Nouvel Identifiant")
                new_pwd = st.text_input("Nouveau Mot de passe", type="password")
                submit_reg = st.form_submit_button("Créer le compte", use_container_width=True)
                
                if submit_reg:
                    if new_user and new_pwd:
                        if create_user(new_user, new_pwd):
                            st.success("✅ Compte créé ! Connectez-vous maintenant.")
                        else:
                            st.error("❌ Ce nom d'utilisateur est déjà pris.")
                    else:
                        st.warning("⚠️ Remplissez tous les champs.")

# --- FONCTION 2 : LE DASHBOARD (Une fois connecté) ---
def main_dashboard():
    # Barre latérale
    with st.sidebar:
        st.title(f"👤 {st.session_state['username']}")
        st.write("Statut : Connecté ✅")
        if st.button("Se déconnecter", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()
        
        st.markdown("---")
        st.info("Données en temps réel issues de l'ETL et de l'IA.")

    # Contenu Principal
    st.title("📊 Tableau de Bord - Prédictions IA")
    st.write("Visualisation des performances des puits (Réel vs Prédictions).")
    st.markdown("---")

    # --- RECUPERATION DES IMAGES ---
    # Chemin : src/App -> src -> PetroScope -> AI_Model -> AI_plots
    project_root = os.path.dirname(os.path.dirname(current_dir))
    plots_dir = os.path.join(project_root, "AI_Model", "AI_plots")
    
    if os.path.exists(plots_dir):
        # On liste les images PNG
        images = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
        
        if images:
            # On trie par date (les plus récentes en premier)
            images.sort(key=lambda x: os.path.getmtime(os.path.join(plots_dir, x)), reverse=True)
            
            st.subheader(f"📸 Derniers Graphiques Générés ({len(images)})")
            
            # Affichage en grille (2 colonnes)
            cols = st.columns(2)
            for idx, img_file in enumerate(images):
                with cols[idx % 2]:
                    st.image(os.path.join(plots_dir, img_file), caption=img_file, use_column_width=True)
                    st.write("") # Espace
        else:
            st.warning("⚠️ Aucune image trouvée dans le dossier AI_plots.")
            st.write(f"Dossier scanné : {plots_dir}")
    else:
        st.error(f"❌ Le dossier des plots n'existe pas : {plots_dir}")

# --- LANCEMENT ---
if st.session_state['logged_in']:
    main_dashboard()
else:
    login_page()