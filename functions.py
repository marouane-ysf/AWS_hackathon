import streamlit as st
import asyncio
import boto3
from botocore.config import Config
import os
import uuid
import json
import time
import PyPDF2
from pypdf import PdfReader
import fitz

# Récupération des IDs des agents à partir de st.secrets
try:
    AGENT_IDS = {
        "manager": st.secrets["bedrock"]["MANAGER_AGENT_ID"],
        "router": st.secrets["bedrock"]["ROUTER_AGENT_ID"],
        "quality": st.secrets["bedrock"]["QUALITY_AGENT_ID"],
        "drafter": st.secrets["bedrock"]["DRAFT_AGENT_ID"],
        "contracts_compare": st.secrets["bedrock"]["COMPARE_AGENT_ID"],
        "market_comparison": st.secrets["bedrock"]["MarketComparisonAgent_ID"],
        "negotiation": st.secrets["bedrock"]["NegotiationAgent_ID"],
        "index_search": st.secrets["bedrock"]["INDEX_SEARCH_AGENT_ID"]
    }

    # Récupération des Alias IDs
    AGENT_ALIAS_IDS = {
        "manager": st.secrets["bedrock"]["MANAGER_AGENT_ALIAS_ID"],
        "router": st.secrets["bedrock"]["ROUTER_AGENT_ALIAS_ID"],
        "quality": st.secrets["bedrock"]["QUALITY_AGENT_ALIAS_ID"],
        "drafter": st.secrets["bedrock"]["DRAFT_AGENT_ALIAS_ID"],
        "contracts_compare": st.secrets["bedrock"]["COMPARE_AGENT_ALIAS_ID"],
        "market_comparison": st.secrets["bedrock"]["MarketComparisonAgent_ALIAS_ID"],
        "negotiation": st.secrets["bedrock"]["NegotiationAgent_ALIAS_ID"],
        "index_search": st.secrets["bedrock"]["INDEX_SEARCH_AGENT_ALIAS_ID"]
    }

    # Configuration Bedrock globale
    SESSION_ID = st.secrets["bedrock"].get("SESSION_ID", "session-capgemini-ai")
    REGION_NAME = st.secrets["aws"]["region"]

except Exception as e:
    # Valeurs par défaut si secrets non configurés
    st.error("Configuration des secrets manquante. Veuillez configurer les secrets dans Streamlit Cloud.")
    AGENT_IDS = {}
    AGENT_ALIAS_IDS = {}
    SESSION_ID = "session-capgemini-ai"
    REGION_NAME = "us-east-1"

# Définition des agents avec leurs informations
AGENTS = {
    "manager": {"name": "Manager Agent", "icon": "🧭", "description": "Répond à des questions d'ordre générale sur le management de contrat"},
    "router": {"name": "Agent Routeur (Orchestration Multi-Agent)", "icon": "🎯", "description": "Orchestrateur intelligent avec accès à tous les agents collaborateurs"},
    "quality": {"name": "Agent Qualité", "icon": "🔍", "description": "Évalue la qualité des contrats et identifie les erreurs clés"},
    "drafter": {"name": "Agent Rédacteur", "icon": "📝", "description": "Prépare des ébauches structurées"},
    "contracts_compare": {"name": "Agent Comparaison de Contrats", "icon": "⚖️", "description": "Compare les contrats et fournit un tableau de comparaison"},
    "market_comparison": {"name": "Agent Comparaison de Marché", "icon": "📊", "description": "Compare différentes options de marché et fournit des insights"},
    "negotiation": {"name": "Agent Négociation", "icon": "🤝", "description": "Assiste dans les stratégies et tactiques de négociation"},
    "index_search": {"name": "Agent Recherche Index", "icon": "🔎", "description": "Recherche dans la base de données Azure Index pour trouver des modèles et contrats"}
}

@st.cache_resource
def get_bedrock_client():
    """Initialise et retourne le client Bedrock avec credentials explicites"""
    try:
        # Configuration pour gérer les timeouts longs
        config = Config(
            read_timeout=3600,  # 1 heure pour les réponses longues
            connect_timeout=60,
            retries={'mode': 'standard', 'max_attempts': 3}
        )
        
        # Créer le client avec les credentials depuis st.secrets
        return boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=st.secrets["aws"]["access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
            region_name=REGION_NAME,
            config=config
        )
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du client Bedrock: {str(e)}")
        return None

def get_or_create_session_id():
    """Génère un session ID unique pour maintenir la cohérence"""
    if "bedrock_session_id" not in st.session_state:
        st.session_state.bedrock_session_id = f"{SESSION_ID}-{uuid.uuid4().hex[:8]}"
    return st.session_state.bedrock_session_id

def validate_agent_configuration():
    """Valide que tous les agents ont leurs IDs configurés"""
    missing_agents = []
    for agent_key in AGENTS.keys():
        if agent_key not in AGENT_IDS or not AGENT_IDS[agent_key] or agent_key not in AGENT_ALIAS_IDS or not AGENT_ALIAS_IDS[agent_key]:
            missing_agents.append(agent_key)
    
    if missing_agents:
        st.error(f"❌ Agents manquants dans la configuration: {missing_agents}")
        st.error("Vérifiez votre configuration des secrets")
        return False
    else:
        return True

# Fonction de test de connexion pour l'agent routeur
async def test_router_connection():
    """Teste la connexion à l'agent routeur et sa capacité de collaboration"""
    try:
        client = get_bedrock_client()
        if not client:
            return {
                "success": False,
                "error": "Impossible d'initialiser le client Bedrock",
                "solution": "Vérifiez vos credentials AWS dans les secrets Streamlit"
            }
            
        current_session_id = get_or_create_session_id()
        
        response = client.invoke_agent(
            agentId=AGENT_IDS["router"],
            agentAliasId=AGENT_ALIAS_IDS["router"],
            sessionId=current_session_id,
            inputText="Test simple: Please respond that you are operational and ready for multi-agent collaboration.",
            enableTrace=True  # Important pour voir les traces de collaboration
        )

        # Extraire la réponse en gérant les traces multi-agent
        full_response = ""
        
        for event in response["completion"]:
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    try:
                        chunk_data = json.loads(chunk["bytes"].decode('utf-8'))
                        if "text" in chunk_data:
                            full_response += chunk_data["text"]
                    except json.JSONDecodeError:
                        full_response += chunk["bytes"].decode('utf-8')
                elif "text" in chunk:
                    full_response += chunk["text"]

        return {
            "success": True,
            "response": full_response.strip(),
            "note": "Agent routeur opérationnel. La collaboration multi-agent se fait automatiquement."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "solution": "Vérifiez que l'agent routeur est configuré avec 'Multi-agent collaboration' activé dans Bedrock."
        }

# FONCTION CORRIGÉE pour parser les traces multi-agent
def parse_multi_agent_traces(response):
    """
    Parse les traces de collaboration multi-agent pour extraire les vraies réponses
    """
    collaborator_responses = []
    final_response = ""
    trace_info = []
    
    for event in response.get("completion", []):
        # Traiter les traces (contiennent les réponses des agents collaborateurs)
        if "trace" in event:
            trace = event["trace"]
            if "trace" in trace:
                trace_data = trace["trace"]
                
                # Vérifier les traces d'orchestration
                if "orchestrationTrace" in trace_data:
                    orch_trace = trace_data["orchestrationTrace"]
                    
                    # Chercher les observations (réponses des agents collaborateurs)
                    if "observation" in orch_trace:
                        observation = orch_trace["observation"]
                        
                        # Si c'est une réponse d'agent collaborateur
                        if observation.get("type") == "AGENT_COLLABORATOR" and "agentCollaboratorInvocationOutput" in observation:
                            collab_output = observation["agentCollaboratorInvocationOutput"]
                            collab_name = collab_output.get("agentCollaboratorName", "Unknown")
                            
                            # Extraire la vraie réponse de l'agent collaborateur
                            if "output" in collab_output and "text" in collab_output["output"]:
                                response_text = collab_output["output"]["text"]
                                collaborator_responses.append({
                                    "agent": collab_name,
                                    "response": response_text
                                })
                                trace_info.append(f"✅ Réponse de {collab_name}: {response_text[:100]}...")
                        
                        # Si c'est la réponse finale
                        elif observation.get("type") == "FINISH" and "finalResponse" in observation:
                            if "text" in observation["finalResponse"]:
                                final_response = observation["finalResponse"]["text"]
        
        # Traiter les chunks (réponse finale consolidée)
        elif "chunk" in event:
            chunk = event["chunk"]
            if "bytes" in chunk:
                try:
                    chunk_data = json.loads(chunk["bytes"].decode('utf-8'))
                    if "text" in chunk_data:
                        final_response += chunk_data["text"]
                except json.JSONDecodeError:
                    chunk_text = chunk["bytes"].decode('utf-8')
                    if chunk_text.strip():
                        final_response += chunk_text
            elif "text" in chunk:
                final_response += chunk["text"]
    
    return {
        "collaborator_responses": collaborator_responses,
        "final_response": final_response.strip(),
        "trace_info": trace_info
    }

# FONCTION PRINCIPALE CORRIGÉE pour gérer multi-agent collaboration
async def execute_agent(agent_key, agent_info, message_content):
    """
    Exécute un agent spécifique avec Bedrock - Gestion spéciale pour multi-agent
    """
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            agent_icon = agent_info['icon']
            agent_name = agent_info['name']
            st.session_state.progress_text = f"{agent_icon} {agent_name}: Traitement en cours..."

            client = get_bedrock_client()
            if not client:
                return f"Erreur: Impossible d'initialiser le client Bedrock pour {agent_name}"
                
            current_session_id = get_or_create_session_id()
            
            # Attendre un peu entre les requêtes pour éviter le throttling
            if attempt > 0:
                time.sleep(retry_delay * attempt)
            
            # Invoquer l'agent avec traces activées
            response = client.invoke_agent(
                agentId=AGENT_IDS[agent_key],
                agentAliasId=AGENT_ALIAS_IDS[agent_key],
                sessionId=current_session_id,
                inputText=message_content,
                enableTrace=True  # CRITIQUE pour multi-agent collaboration
            )

            # Si c'est l'agent routeur, parser spécialement les traces multi-agent
            if agent_key == "router":
                parsed_response = parse_multi_agent_traces(response)
                
                # Debug info si activé
                if st.session_state.debug_mode:
                    if parsed_response["trace_info"]:
                        st.info("🔍 Traces de collaboration multi-agent:")
                        for trace in parsed_response["trace_info"]:
                            st.write(f"  - {trace}")
                
                # Construire la réponse finale
                if parsed_response["collaborator_responses"]:
                    # S'il y a des réponses d'agents collaborateurs
                    responses_text = []
                    for collab in parsed_response["collaborator_responses"]:
                        responses_text.append(f"**{collab['agent']}**:\n{collab['response']}")
                    
                    # Si pas de réponse finale consolidée, utiliser les réponses des collaborateurs
                    if not parsed_response["final_response"]:
                        return "\n\n".join(responses_text)
                    else:
                        # Sinon retourner la réponse finale consolidée
                        return parsed_response["final_response"]
                else:
                    # Si pas de réponses de collaborateurs, retourner la réponse finale
                    return parsed_response["final_response"] if parsed_response["final_response"] else f"Pas de réponse de {agent_name}"
            
            else:
                # Pour les autres agents (non-routeur), traitement standard
                full_response = ""
                for event in response["completion"]:
                    if "chunk" in event:
                        chunk = event["chunk"]
                        if "bytes" in chunk:
                            try:
                                chunk_data = json.loads(chunk["bytes"].decode('utf-8'))
                                if "text" in chunk_data:
                                    full_response += chunk_data["text"]
                            except json.JSONDecodeError:
                                full_response += chunk["bytes"].decode('utf-8')
                        elif "text" in chunk:
                            full_response += chunk["text"]

                return full_response.strip() if full_response.strip() else f"Pas de réponse de {agent_name}"

        except Exception as e:
            if "throttling" in str(e).lower() and attempt < max_retries - 1:
                st.warning(f"Ralentissement Bedrock, nouvel essai dans {retry_delay * (attempt + 1)} secondes...")
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                error_msg = f"Erreur lors de l'exécution de {agent_info['name']}: {e}"
                st.error(error_msg)
                return error_msg

# Fonction pour exécuter un pipeline séquentiel
async def run_sequential_pipeline(query):
    """Exécute un pipeline séquentiel avec les agents définis par l'utilisateur"""
    try:
        sequence = st.session_state.get("agent_sequence", [])
        if not sequence:
            return {"error": "Aucune séquence d'agents définie. Veuillez définir une séquence dans la barre latérale."}

        responses = {}
        current_input = query

        for i, agent_key in enumerate(sequence):
            try:
                agent_info = AGENTS[agent_key]
                st.session_state.progress_text = f"{agent_info['icon']} {agent_info['name']}: Traitement en cours..."
                st.session_state.progress_value = (i + 1) / len(sequence)

                response = await execute_agent(agent_key, agent_info, current_input)
                responses[agent_key] = response
                current_input = f"Tenant compte de la réponse précédente: {response}\n\nQuestion initiale: {query}"
            
            except Exception as agent_error:
                error_message = f"Erreur: {str(agent_error)}"
                responses[agent_key] = error_message
                current_input = f"L'agent précédent a rencontré une erreur. Question initiale: {query}"

        st.session_state.progress_text = "✅ Traitement terminé"
        st.session_state.progress_value = 1.0

        combined_response = "\n\n".join(f"{AGENTS[agent_key]['icon']} {AGENTS[agent_key]['name']}:\n{response}" for agent_key, response in responses.items())

        return {
            "selected_agents": sequence,
            "agent_names": [AGENTS[agent]['name'] for agent in sequence],
            "agent_icons": [AGENTS[agent]['icon'] for agent in sequence],
            "combined": combined_response,
            **responses
        }

    except Exception as e:
        return {"error": f"Erreur lors de l'exécution du workflow multi-agent: {str(e)}"}

# Fonction pour exécuter un agent spécifique (mode agent unique)
async def run_specific_agent(query, agent_key):
    """Exécute un agent spécifique (mode agent unique)"""
    try:
        agent_name = AGENTS[agent_key]["name"]
        agent_icon = AGENTS[agent_key]["icon"]

        st.session_state.progress_text = f"{agent_icon} {agent_name}: Préparation de votre réponse..."
        st.session_state.progress_value = 0.5

        response = await execute_agent(agent_key, AGENTS[agent_key], query)

        st.session_state.progress_text = "✅ Traitement terminé"
        st.session_state.progress_value = 1.0

        return {
            "selected_agent": agent_key,
            "agent_name": agent_name,
            "agent_icon": agent_icon,
            "combined": response
        }

    except Exception as e:
        return {"error": f"Erreur lors de l'exécution de l'agent {agent_key}: {str(e)}"}

# FONCTION PRINCIPALE SIMPLIFIÉE
async def run_workflow_based_on_mode(query, mode):
    """
    Version SIMPLIFIÉE - L'agent router gère automatiquement la collaboration
    """
    if mode == "intelligent":
        st.session_state.progress_text = f"🎯 Agent Routeur: Orchestration multi-agent en cours..."
        
        # TRAITEMENT IDENTIQUE - Le router gère automatiquement ses collaborateurs
        response = await run_specific_agent(query, "router")
        
        # Ajouter des informations spécifiques au routeur
        response["selection_method"] = "Orchestration Multi-Agent Automatique"
        return response
        
    elif mode == "sequence":
        return await run_sequential_pipeline(query)
    else:
        if st.session_state.selected_agents and all(agent in AGENTS for agent in st.session_state.selected_agents):
            return await run_specific_agent(query, st.session_state.selected_agents[0])
        else:
            return {"error": "Veuillez sélectionner un agent dans la barre latérale pour continuer."}

# Fonction pour exécuter les fonctions asynchrones dans Streamlit
def run_async_function(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()

# Fonctions pour extraction de texte PDF (inchangées)
def extract_text_from_pdf_ocr(pdf_document):
    """for OCR"""
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_pdf(uploaded_file, ocr):
    if (uploaded_file is not None) and ocr:
        file_name = uploaded_file.name
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted_text = extract_text_from_pdf_ocr(pdf_document)
        raw_text = extracted_text
        return raw_text, file_name
    else:
        if uploaded_file is not None:
            file_name = uploaded_file.name
            if uploaded_file.type == "text/plain":
                raw_text = str(uploaded_file.read(),"utf-8")
            elif uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                raw_text = text
            else:
                raw_text = ""
                file_name = None
        else:
            raw_text = ""
            file_name = None
    return raw_text, file_name

def extract_text_from_multiple_files(uploaded_files, ocr):
    """extract text from multiple files and return a list of file text"""
    files_text = []
    progress_bar = st.progress(0)

    total_files = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(i / total_files)
        file_content, file_name = extract_text_from_pdf(uploaded_file, ocr)
        if file_name:
            files_text.append({
                'content': file_content,
                'name': file_name
            })
            st.write(f"✅ {uploaded_file.name} content extracted successfully")
        else:
            st.error(f"❌ Failed to extract content from {uploaded_file.name}")

    progress_bar.progress(1.0)
    return files_text

def prompt_constructor(user_input, ocr):
    """Construit le prompt avec gestion des fichiers"""
    # Gestion du cas où user_input est une string simple (depuis chat_input)
    if isinstance(user_input, str):
        return user_input
    
    # Gestion du cas où user_input est un dictionnaire (avec fichiers)
    msg = user_input.get("text", "")
    files = user_input.get("files", [])
    
    if files:
        if msg is None or msg == "":
            msg = "sharing documents"
        files_content = extract_text_from_multiple_files(files, ocr)
        user_prompt = msg
        for i, file in enumerate(files_content):
            user_prompt += f"\ncontract n°{i+1} called " + file["name"] + "\n" + file["content"]
            if "uploaded_file" in st.session_state and isinstance(st.session_state.uploaded_file, list):
                st.session_state.uploaded_file.append(file)
            else:
                st.session_state.uploaded_file = [file]
    else:
        user_prompt = msg
    
    return user_prompt
