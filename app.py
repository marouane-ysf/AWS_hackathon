import streamlit as st
from functions import *
import os
from pathlib import Path

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Capgemini AI Multi-Agent System",
    page_icon="☘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
        text-align: center;
        margin-bottom: 1rem;
    }
    .agent-card {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .agent-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .debug-info {
        background-color: #fff8e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        border-left: 3px solid #ffc107;
        font-size: 0.8rem;
    }
    .router-response {
        font-family: monospace;
        background-color: #f1f1f1;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }
    .context-info {
        background-color: #e8f4ff;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        border-left: 3px solid #0066cc;
    }
    .orchestration-info {
        background-color: #e8f8e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .router-selected {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
    }
    .test-info {
        background-color: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #6c63ff;
    }
    .selected-agent {
        border-left: 5px solid #003366;
        font-weight: 600;
        background-color: #f0f8ff;
    }
    
    /* Masquer le footer par défaut de Streamlit */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="main-header">Capgemini AI Multi-Agent System</div>', unsafe_allow_html=True)

# Initialisation des variables de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "orchestration_mode" not in st.session_state:
    st.session_state.orchestration_mode = "intelligent"
if "selected_agents" not in st.session_state:
    st.session_state.selected_agents = []
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "router_raw_response" not in st.session_state:
    st.session_state.router_raw_response = ""
if "agent_sequence" not in st.session_state:
    st.session_state.agent_sequence = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = []
if "context_mode" not in st.session_state:
    st.session_state.context_mode = True
if "progress_text" not in st.session_state:
    st.session_state.progress_text = ""
if "progress_value" not in st.session_state:
    st.session_state.progress_value = 0.0

# Barre latérale pour la configuration
with st.sidebar:
    # Affichage du logo (avec fallback)
    try:
        current_dir = Path(__file__).parent
        image_path = os.path.join(current_dir, "assets", "capgemini.png")
        if os.path.exists(image_path):
            st.image(image_path, width=190)
        else:
            # Fallback if image can't be loaded
            st.markdown("### Capgemini AI")
    except Exception:
        # Fallback if image can't be loaded
        st.markdown("### Capgemini AI")
    
    # Validation de la configuration Bedrock
    missing_agents = []
    for agent_key in AGENTS.keys():
        if agent_key not in AGENT_IDS or not AGENT_IDS[agent_key] or agent_key not in AGENT_ALIAS_IDS or not AGENT_ALIAS_IDS[agent_key]:
            missing_agents.append(agent_key)
    
    if missing_agents:
        st.error(f"⚠️ Configuration manquante pour: {', '.join(missing_agents)}")
        st.info("Vérifiez votre configuration")
    else:
        st.success("✅ Tous les agents Bedrock sont configurés")
    
    # NOUVEAU : Bouton de test de connexion
    st.markdown("### 🔧 Diagnostic")
    if st.button("🧪 Tester l'Agent Routeur", help="Teste la connexion à l'agent routeur"):
        with st.spinner("Test de connexion en cours..."):
            test_result = run_async_function(test_router_connection)
            
            if test_result["success"]:
                st.success("✅ Agent Routeur opérationnel!")
                if test_result["response"]:
                    st.markdown(f"""
                    <div class="test-info">
                        <b>Réponse de test:</b><br>
                        {test_result["response"][:200]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Erreur de connexion: {test_result['error']}")
    
    # Sélection du mode d'orchestration - SIMPLIFIÉ
    st.markdown("### 🎯 Mode de Fonctionnement")
    mode = st.radio(
        "Choisissez un mode:",
        ["Orchestration Intelligente", "Séquence Multi-Agent", "Agent Unique"],
        index=0,
        help="• Orchestration Intelligente: Agent Routeur avec orchestration automatique\n"
             "• Séquence Multi-Agent: Vous définissez l'ordre des agents\n"
             "• Agent Unique: Sélectionnez un agent spécifique"
    )

    if mode == "Orchestration Intelligente":
        st.session_state.orchestration_mode = "intelligent"
        st.markdown("""
        <div class="router-selected">
            🎯 <strong>Agent Routeur Sélectionné</strong><br>
            L'agent routeur gérera automatiquement l'orchestration avec ses agents collaborateurs configurés dans Bedrock.
        </div>
        """, unsafe_allow_html=True)
         
    elif mode == "Séquence Multi-Agent":
        st.session_state.orchestration_mode = "sequence"
    else:
        st.session_state.orchestration_mode = "single"

    # Interface pour définir la séquence personnalisée
    if st.session_state.orchestration_mode == "sequence":
        st.markdown("### 📋 Définir la séquence d'agents")
        sequence = []
        for agent_key, agent_info in AGENTS.items():
            if agent_key != "router":  
                if st.checkbox(f"{agent_info['icon']} {agent_info['name']}", key=f"seq_{agent_key}"):
                    sequence.append(agent_key)

        # Permettre à l'utilisateur de définir l'ordre
        if sequence:
            sequence = st.multiselect(
                "Définissez l'ordre des agents:",
                options=sequence,
                default=sequence,
                key="agent_sequence_select"
            )
            st.session_state.agent_sequence = sequence

    # Si mode agent unique, sélecteur d'agent
    if st.session_state.orchestration_mode == "single":
        st.markdown("### 🎯 Sélection d'agent unique")
        for agent_key, agent_info in AGENTS.items():
            if agent_key != "router":
                if st.button(f"{agent_info['icon']} {agent_info['name']}", help=agent_info['description'], key=f"btn_{agent_key}"):
                    st.session_state.selected_agents = [agent_key]

    # Options de configuration
    st.markdown("### ⚙️ Configuration")
    st.session_state.context_mode = st.checkbox("Maintenir le contexte", value=True, 
                                              help="Active/désactive la mémoire des conversations précédentes")
    
    st.session_state.debug_mode = st.checkbox("Mode debug", value=st.session_state.debug_mode,
                                            help="Affiche des informations détaillées sur le traitement")
    
    # Option pour forcer le mode direct du routeur
    if st.session_state.orchestration_mode == "intelligent":
        if "direct_mode" not in st.session_state:
            st.session_state.direct_mode = True
        
        st.session_state.direct_mode = st.checkbox("Mode réponse directe", value=True,
                                                 help="Force l'agent routeur à répondre directement sans JSON d'orchestration")
        
        if st.session_state.direct_mode:
            st.markdown("""
            <div class="context-info">
                Mode direct activé. L'agent routeur donnera des réponses finales directement.
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.context_mode:
        st.markdown("""
        <div class="context-info">
            Mode contexte activé. Les agents se souviendront des interactions précédentes.
        </div>
        """, unsafe_allow_html=True)
    
    # Affichage du Session ID actuel si en mode debug
    if st.session_state.debug_mode and "bedrock_session_id" in st.session_state:
        st.markdown(f"""
        <div class="debug-info">
            <b>Session ID actuel:</b><br>
            {st.session_state.bedrock_session_id}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 Agents disponibles")
    for agent_key, agent_info in AGENTS.items():
       css_class = "agent-card"
    
    # Logique d'affichage selon le mode
       if st.session_state.orchestration_mode == "intelligent" and agent_key == "router":
           css_class += " selected-agent"
       elif st.session_state.orchestration_mode != "intelligent" and agent_key != "router" and agent_key in st.session_state.selected_agents:
           css_class += " selected-agent"
    
    # Afficher le statut de configuration
       config_status = "✅" if (agent_key in AGENT_IDS and AGENT_IDS[agent_key] and agent_key in AGENT_ALIAS_IDS and AGENT_ALIAS_IDS[agent_key]) else "❌"

    # CORRECTION: En mode single, ne pas afficher le router
       if st.session_state.orchestration_mode == "single" and agent_key == "router":
          continue
        
       st.markdown(f"""
       <div class="{css_class}">
           <div class="agent-header">{config_status} {agent_info['icon']} {agent_info['name']}</div>
           <div>{agent_info['description']}</div>
       </div>
       """, unsafe_allow_html=True)

    if st.button("🔄 Réinitialiser la conversation", help="Effacer l'historique de conversation"):
        st.session_state.messages = []
        st.session_state.current_results = None
        st.session_state.agent_sequence = []
        st.session_state.selected_agents = []
        st.session_state.uploaded_file = []
        # Réinitialiser le session ID Bedrock
        if "bedrock_session_id" in st.session_state:
            del st.session_state.bedrock_session_id
        st.rerun()

# Checkbox pour activer l'OCR
ocr1 = st.checkbox("Check the box to enable OCR to read scanned pdf that are images", key="ocr1")

# Affichage de l'historique des messages en utilisant st.chat_message
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        # Déterminer l'icône et le nom de l'agent pour l'affichage
        agent_prefix = ""
        if "agent_icon" in message and "agent_name" in message:
            agent_prefix = f"{message['agent_icon']} **{message['agent_name']}**:\n\n"
        elif "agent_icons" in message and "agent_names" in message:
            agent_prefix = f"{', '.join(message['agent_icons'])} **{', '.join(message['agent_names'])}**:\n\n"
        
        with st.chat_message("assistant"):
            if agent_prefix:
                st.markdown(agent_prefix)
            st.write(message["content"])
            
            # Afficher les informations de debug si nécessaire
            if st.session_state.debug_mode and "selection_method" in message:
                st.markdown(f"""
                <div class="debug-info">
                    <b>Méthode de sélection:</b> {message["selection_method"]}<br>
                    <b>Informations routeur:</b>
                    <div class="router-response">{message.get("router_response", "Non disponible")}</div>
                </div>
                """, unsafe_allow_html=True)

# Afficher la barre de progression si nécessaire
if st.session_state.processing:
    st.markdown(f"<div style='margin-top: 1rem;'>{st.session_state.progress_text}</div>", unsafe_allow_html=True)
    st.progress(st.session_state.progress_value)

# Section pour afficher les réponses détaillées - SEULEMENT pour les séquences
if st.session_state.current_results and "error" not in st.session_state.current_results:
    if st.session_state.orchestration_mode == "sequence" and "selected_agents" in st.session_state.current_results:
        selected_agents = st.session_state.current_results["selected_agents"]
        if len(selected_agents) > 1:
            with st.expander("📊 Voir les réponses détaillées de chaque agent"):
                cols = st.columns(len(selected_agents))
                
                for i, agent_key in enumerate(selected_agents):
                    if agent_key in st.session_state.current_results:
                        with cols[i]:
                            st.markdown(f"""
                            <div class="agent-card">
                                <div class="agent-header">{AGENTS[agent_key]['icon']} {AGENTS[agent_key]['name']}</div>
                                <div style="max-height: 300px; overflow-y: auto;">
                                    {st.session_state.current_results[agent_key]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

# Chat input utilisant le composant natif de Streamlit
user_prompt = st.chat_input("Tapez votre message ici...", disabled=st.session_state.processing)

if user_prompt:
    # Gérer les fichiers uploadés via st.file_uploader si nécessaire
    uploaded_files = st.file_uploader("Télécharger des fichiers", accept_multiple_files=True, key="file_uploader")
    
    # Créer un dictionnaire pour simuler la structure attendue
    user_input_dict = {
        "text": user_prompt,
        "files": uploaded_files if uploaded_files else []
    }
    
    user_input = prompt_constructor(user_input_dict, ocr1)

    if user_input and not st.session_state.processing:
        # Afficher le message utilisateur en utilisant st.chat_message
        with st.chat_message("user"):
            st.write(user_prompt)

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.processing = True
        st.session_state.progress_text = "Initialisation du traitement..."
        st.session_state.progress_value = 0.1

        try:
            # Utiliser la nouvelle fonction de workflow SIMPLIFIÉE
            if st.session_state.orchestration_mode == "intelligent":
                with st.spinner("🎯 Agent Routeur en cours d'orchestration..."):
                    result = run_async_function(run_workflow_based_on_mode, user_input, "intelligent")
            elif st.session_state.orchestration_mode == "sequence":
                with st.spinner("🔄 Les agents collaborent en séquence pour répondre à votre question..."):
                    result = run_async_function(run_workflow_based_on_mode, user_input, "sequence")
            else:
                if st.session_state.selected_agents and all(agent in AGENTS for agent in st.session_state.selected_agents):
                    agent_name = AGENTS[st.session_state.selected_agents[0]]['name']
                    with st.spinner(f"🤖 {agent_name} prépare votre réponse..."):
                        result = run_async_function(run_workflow_based_on_mode, user_input, "single")
                else:
                    result = {"error": "Veuillez sélectionner un agent dans la barre latérale pour continuer."}

            st.session_state.processing = False

            if "error" in result:
                st.error(result["error"])
                st.session_state.messages.append({"role": "assistant", "content": result["error"]})
                st.rerun()
            else:
                st.session_state.current_results = result

                # Préparer les informations d'agent pour l'affichage dans le message
                agent_prefix = ""
                
                if "agent_names" in result and "agent_icons" in result:
                    agent_prefix = f"{', '.join(result['agent_icons'])} **{', '.join(result['agent_names'])}**:\n\n"
                elif "agent_name" in result and "agent_icon" in result:
                    agent_prefix = f"{result['agent_icon']} **{result['agent_name']}**:\n\n"
                
                # Afficher la réponse de l'assistant
                with st.chat_message("assistant"):
                    if agent_prefix:
                        st.markdown(agent_prefix)
                    st.write(result["combined"])
                    
                    # Afficher les informations de debug si nécessaire
                    if st.session_state.debug_mode and "selection_method" in result:
                        st.markdown(f"""
                        <div class="debug-info">
                            <b>Méthode de sélection:</b> {result["selection_method"]}<br>
                            <b>Informations:</b>
                            <div class="router-response">{result.get("router_response", "Non disponible")}</div>
                        </div>
                        """, unsafe_allow_html=True)

                message_data = {
                    "role": "assistant",
                    "content": result["combined"]
                }

                if "agent_names" in result:
                    message_data["agent_names"] = result["agent_names"]
                    message_data["agent_icons"] = result["agent_icons"]
                elif "agent_name" in result: 
                    message_data["agent_name"] = result["agent_name"]
                    message_data["agent_icon"] = result["agent_icon"]

                if "selection_method" in result:
                    message_data["selection_method"] = result["selection_method"]
                if "router_response" in result:
                    message_data["router_response"] = result["router_response"]

                st.session_state.messages.append(message_data)
                st.rerun()  # Rafraîchir l'interface pour afficher le nouveau message

        except Exception as e:
            st.session_state.processing = False
            st.error(f"Erreur lors du traitement: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Erreur lors du traitement: {str(e)}"})
            st.rerun()

# Pied de page
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.8rem;">
    <p>Développé pour Capgemini AI Agents - Système d'Orchestration Intelligente | 2025 - Powered by Amazon Bedrock</p>
    <p>🎯 Agent Routeur avec Multi-Agent Collaboration Native</p>
</div>
""", unsafe_allow_html=True)