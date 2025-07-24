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
from typing import Dict, List, Optional, Tuple
import re

# Tentative d'import de fitz, mais pas critique si ça échoue
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

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
        # Configuration pour gérer les timeouts longs et le streaming
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

        # Parser la réponse avec la nouvelle méthode
        parsed_response = parse_multi_agent_response_complete(response)
        
        return {
            "success": True,
            "response": parsed_response["final_response"],
            "note": "Agent routeur opérationnel. La collaboration multi-agent se fait automatiquement."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "solution": "Vérifiez que l'agent routeur est configuré avec 'Multi-agent collaboration' activé dans Bedrock."
        }

# NOUVELLE FONCTION COMPLÈTE pour parser les réponses multi-agent
def parse_multi_agent_response_complete(response: Dict) -> Dict:
    """
    Parse complète des réponses multi-agent basée sur les patterns AWS officiels
    Gère les traces, les réponses des collaborateurs et les erreurs RerunData
    """
    result = {
        "final_response": "",
        "collaborator_responses": {},
        "orchestration_steps": [],
        "trace_info": [],
        "errors": []
    }
    
    try:
        # Parcourir tous les événements dans la réponse
        for event in response.get("completion", []):
            
            # 1. Traiter les chunks (réponse finale)
            if "chunk" in event and event["chunk"]:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    decoded_bytes = chunk["bytes"].decode('utf-8')
                    
                    # Filtrer les erreurs RerunData
                    if "RerunData" in decoded_bytes:
                        result["errors"].append("RerunData error detected and filtered")
                        continue
                    
                    # Essayer de parser comme JSON d'abord
                    try:
                        chunk_data = json.loads(decoded_bytes)
                        if "text" in chunk_data:
                            result["final_response"] += chunk_data["text"]
                    except json.JSONDecodeError:
                        # Si ce n'est pas du JSON, ajouter tel quel
                        if decoded_bytes.strip():
                            result["final_response"] += decoded_bytes
            
            # 2. Traiter les traces (contiennent les réponses des collaborateurs)
            elif "trace" in event:
                trace_part = event["trace"]
                
                # Extraire les informations du collaborateur
                collaborator_name = trace_part.get("collaboratorName")
                caller_chain = trace_part.get("callerChain", [])
                
                # Si on a un nom de collaborateur, l'ajouter aux infos
                if collaborator_name:
                    result["trace_info"].append(f"Trace from: {collaborator_name}")
                
                # Parser le contenu de la trace
                if "trace" in trace_part:
                    trace_content = trace_part["trace"]
                    
                    # Traiter les traces de pré-processing
                    if "preProcessingTrace" in trace_content:
                        pre_trace = trace_content["preProcessingTrace"]
                        if "modelInvocationInput" in pre_trace:
                            input_text = pre_trace["modelInvocationInput"].get("text", "")
                            result["orchestration_steps"].append({
                                "type": "pre_processing",
                                "input": input_text[:200] + "..." if len(input_text) > 200 else input_text
                            })
                    
                    # Traiter les traces d'orchestration (CRITIQUE pour multi-agent)
                    if "orchestrationTrace" in trace_content:
                        orch_trace = trace_content["orchestrationTrace"]
                        
                        # Extraire le raisonnement
                        if "modelInvocationInput" in orch_trace:
                            reasoning = orch_trace["modelInvocationInput"].get("text", "")
                            result["orchestration_steps"].append({
                                "type": "orchestration",
                                "reasoning": reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                            })
                        
                        # EXTRAIRE LES RÉPONSES DES AGENTS COLLABORATEURS
                        if "observation" in orch_trace:
                            observation = orch_trace["observation"]
                            obs_type = observation.get("type", "")
                            
                            # Si c'est une réponse d'agent collaborateur
                            if obs_type == "AGENT_COLLABORATOR" and "agentCollaboratorInvocationOutput" in observation:
                                collab_output = observation["agentCollaboratorInvocationOutput"]
                                
                                # Extraire le nom et la réponse du collaborateur
                                collab_name = collab_output.get("agentCollaboratorName", "Unknown")
                                collab_alias = collab_output.get("agentCollaboratorAliasArn", "")
                                
                                # Extraire la réponse textuelle
                                if "output" in collab_output:
                                    output_text = collab_output["output"].get("text", "")
                                    
                                    # Stocker la réponse du collaborateur
                                    result["collaborator_responses"][collab_name] = {
                                        "response": output_text,
                                        "alias": collab_alias
                                    }
                                    
                                    # Ajouter à l'orchestration
                                    result["orchestration_steps"].append({
                                        "type": "collaborator_response",
                                        "agent": collab_name,
                                        "response_preview": output_text[:200] + "..." if len(output_text) > 200 else output_text
                                    })
                            
                            # Si c'est une action group
                            elif obs_type == "ACTION_GROUP" and "actionGroupInvocationOutput" in observation:
                                action_output = observation["actionGroupInvocationOutput"]
                                action_text = action_output.get("text", "")
                                result["orchestration_steps"].append({
                                    "type": "action_group",
                                    "output": action_text[:200] + "..." if len(action_text) > 200 else action_text
                                })
                            
                            # Si c'est une recherche knowledge base
                            elif obs_type == "KNOWLEDGE_BASE" and "knowledgeBaseLookupOutput" in observation:
                                kb_output = observation["knowledgeBaseLookupOutput"]
                                references = kb_output.get("retrievedReferences", [])
                                result["orchestration_steps"].append({
                                    "type": "knowledge_base",
                                    "references_count": len(references)
                                })
                            
                            # Si c'est la réponse finale
                            elif obs_type == "FINISH" and "finalResponse" in observation:
                                final_text = observation["finalResponse"].get("text", "")
                                if final_text and not result["final_response"]:
                                    result["final_response"] = final_text
                    
                    # Traiter les traces post-processing
                    if "postProcessingTrace" in trace_content:
                        post_trace = trace_content["postProcessingTrace"]
                        if "modelInvocationOutput" in post_trace:
                            parsed_response = post_trace["modelInvocationOutput"].get("parsedResponse", {})
                            result["orchestration_steps"].append({
                                "type": "post_processing",
                                "output": str(parsed_response)[:200] + "..." if len(str(parsed_response)) > 200 else str(parsed_response)
                            })
    
    except Exception as e:
        result["errors"].append(f"Erreur lors du parsing: {str(e)}")
        st.error(f"Erreur de parsing des traces: {e}")
    
    # Si pas de réponse finale mais des réponses de collaborateurs, les consolider
    if not result["final_response"] and result["collaborator_responses"]:
        consolidated = []
        for agent_name, agent_data in result["collaborator_responses"].items():
            agent_icon = "🤖"
            # Trouver l'icône correspondante
            for key, info in AGENTS.items():
                if agent_name.lower() in info["name"].lower() or key in agent_name.lower():
                    agent_icon = info["icon"]
                    break
            consolidated.append(f"{agent_icon} **{agent_name}**:\n{agent_data['response']}")
        result["final_response"] = "\n\n".join(consolidated)
    
    return result

# FONCTION PRINCIPALE AMÉLIORÉE pour gérer multi-agent collaboration
async def execute_agent(agent_key, agent_info, message_content):
    """
    Exécute un agent spécifique avec Bedrock - Version complète avec parsing avancé
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
                enableTrace=True,  # CRITIQUE pour multi-agent collaboration
                endSession=False   # Garder la session ouverte
            )

            # Utiliser le nouveau parser complet
            parsed_response = parse_multi_agent_response_complete(response)
            
            # Si mode debug, afficher les détails de l'orchestration
            if st.session_state.debug_mode:
                # Afficher les étapes d'orchestration
                if parsed_response["orchestration_steps"]:
                    st.info("🔍 Étapes d'orchestration:")
                    for step in parsed_response["orchestration_steps"]:
                        if step["type"] == "orchestration":
                            st.write(f"  📋 Raisonnement: {step['reasoning']}")
                        elif step["type"] == "collaborator_response":
                            st.write(f"  ✅ Réponse de {step['agent']}: {step['response_preview']}")
                        elif step["type"] == "action_group":
                            st.write(f"  ⚡ Action: {step['output']}")
                        elif step["type"] == "knowledge_base":
                            st.write(f"  📚 Knowledge Base: {step['references_count']} références trouvées")
                
                # Afficher les erreurs filtrées
                if parsed_response["errors"]:
                    st.warning(f"⚠️ Erreurs filtrées: {', '.join(parsed_response['errors'])}")
            
            # Si c'est l'agent routeur, formater spécialement la réponse
            if agent_key == "router" and parsed_response["collaborator_responses"]:
                # Créer une belle réponse consolidée
                sections = []
                
                # Ajouter un résumé si disponible
                if parsed_response["final_response"]:
                    sections.append("## 📊 Synthèse de l'orchestration\n" + parsed_response["final_response"])
                
                # Ajouter les réponses des collaborateurs
                if parsed_response["collaborator_responses"]:
                    sections.append("\n## 🤝 Réponses des agents collaborateurs")
                    for agent_name, agent_data in parsed_response["collaborator_responses"].items():
                        # Trouver l'icône de l'agent
                        agent_icon = "🤖"
                        for key, info in AGENTS.items():
                            if agent_name.lower() in info["name"].lower() or key in agent_name.lower():
                                agent_icon = info["icon"]
                                break
                        
                        sections.append(f"\n### {agent_icon} {agent_name}\n{agent_data['response']}")
                
                return "\n".join(sections)
            else:
                # Pour les autres agents, retourner simplement la réponse finale
                return parsed_response["final_response"] if parsed_response["final_response"] else f"Pas de réponse de {agent_name}"

        except Exception as e:
            error_str = str(e).lower()
            
            # Gestion spécifique du throttling
            if "throttling" in error_str and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                st.warning(f"⏳ Limite de débit atteinte. Nouvelle tentative dans {wait_time} secondes...")
                await asyncio.sleep(wait_time)
                continue
            
            # Gestion des autres erreurs communes
            elif "accessdenied" in error_str:
                return f"❌ Accès refusé pour {agent_name}. Vérifiez les permissions IAM."
            elif "resourcenotfound" in error_str:
                return f"❌ Agent {agent_name} non trouvé. Vérifiez l'ID et l'alias."
            else:
                error_msg = f"❌ Erreur lors de l'exécution de {agent_info['name']}: {e}"
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
            "combined": response,
            "selection_method": "Agent unique sélectionné manuellement"
        }

    except Exception as e:
        return {"error": f"Erreur lors de l'exécution de l'agent {agent_key}: {str(e)}"}

# FONCTION PRINCIPALE SIMPLIFIÉE
async def run_workflow_based_on_mode(query, mode):
    """
    Version complète avec support multi-agent avancé
    """
    if mode == "intelligent":
        st.session_state.progress_text = f"🎯 Agent Routeur: Orchestration multi-agent en cours..."
        
        # Le router gère automatiquement ses collaborateurs
        response = await run_specific_agent(query, "router")
        
        # Ajouter des informations spécifiques au routeur
        response["selection_method"] = "Orchestration Multi-Agent Automatique"
        response["router_response"] = "Orchestration complète avec collaboration multi-agent"
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

# Fonctions pour extraction de texte PDF MODIFIÉES
def extract_text_from_pdf_ocr(pdf_document):
    """for OCR - utilise fitz si disponible"""
    if FITZ_AVAILABLE:
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    else:
        # Fallback vers pypdf si fitz n'est pas disponible
        return None

def extract_text_from_pdf(uploaded_file, ocr):
    """Extraction de texte avec gestion de fitz optionnel"""
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        # Si OCR demandé ET fitz disponible
        if ocr and FITZ_AVAILABLE:
            try:
                pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                raw_text = extract_text_from_pdf_ocr(pdf_document)
                if raw_text:
                    return raw_text, file_name
            except Exception as e:
                st.warning(f"Erreur OCR avec fitz: {e}. Utilisation de pypdf.")
        
        # Réinitialiser le pointeur du fichier
        uploaded_file.seek(0)
        
        # Utiliser pypdf (toujours disponible)
        if uploaded_file.type == "text/plain":
            raw_text = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            raw_text = text
        else:
            raw_text = ""
            file_name = None
            
        return raw_text, file_name
    else:
        return "", None

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
