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
        # Configuration optimisée pour multi-agent collaboration
        config = Config(
            read_timeout=7200,  # 2 heures pour orchestrations complexes
            connect_timeout=120,
            retries={'mode': 'adaptive', 'max_attempts': 5},
            max_pool_connections=50,
            # Paramètres spécifiques pour Bedrock
            parameter_validation=False,  # Éviter les validations strictes
            signature_version='v4'  # Version de signature AWS
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
        # Session ID plus spécifique pour éviter les conflits
        timestamp = int(time.time())
        st.session_state.bedrock_session_id = f"streamlit-{timestamp}-{uuid.uuid4().hex[:12]}"
    return st.session_state.bedrock_session_id

# FONCTION DE DIAGNOSTIC MULTI-AGENT
async def diagnose_router_agent():
    """Diagnostique l'agent routeur et sa configuration multi-agent"""
    try:
        client = get_bedrock_client()
        if not client:
            return {"error": "Client Bedrock non disponible"}
        
        session_id = get_or_create_session_id()
        
        # Test simple d'orchestration
        test_query = "Please orchestrate with your collaborator agents to provide a brief overview of contract management best practices. This should involve multiple agents working together."
        
        response = client.invoke_agent(
            agentId=AGENT_IDS["router"],
            agentAliasId=AGENT_ALIAS_IDS["router"],
            sessionId=session_id,
            inputText=test_query,
            enableTrace=True,
            endSession=False
        )
        
        parsed = parse_multi_agent_response_complete(response)
        
        # Analyser les résultats
        diagnosis = {
            "router_responsive": bool(parsed["final_response"]),
            "collaboration_detected": bool(parsed["collaborator_responses"]),
            "orchestration_steps": len(parsed["orchestration_steps"]),
            "collaborators_count": len(parsed["collaborator_responses"]),
            "errors": parsed["errors"],
            "raw_response_length": len(parsed["final_response"]),
            "recommendation": ""
        }
        
        # Générer des recommandations
        if not diagnosis["collaboration_detected"]:
            diagnosis["recommendation"] = "L'agent routeur ne collabore pas avec d'autres agents. Vérifiez la configuration 'Multi-agent collaboration' dans AWS Bedrock."
        elif diagnosis["collaborators_count"] < 2:
            diagnosis["recommendation"] = "Collaboration limitée détectée. Vérifiez que tous les agents collaborateurs sont correctement configurés."
        else:
            diagnosis["recommendation"] = "Configuration multi-agent semble correcte."
        
        return diagnosis
        
    except Exception as e:
        return {"error": f"Erreur de diagnostic: {str(e)}"}

def validate_agent_configuration():
    """Validation complète de la configuration des agents"""
    config_status = validate_multi_agent_setup()
    
    if not config_status["valid"]:
        st.error(f"❌ Configuration invalide: {len(config_status['issues'])} problèmes détectés")
        for issue in config_status["issues"]:
            st.error(f"  • {issue}")
        st.info(f"Agents configurés: {config_status['configured_agents']}/{config_status['total_agents']}")
        return False
    else:
        st.success(f"✅ Configuration valide: {config_status['total_agents']} agents configurés")
        return True

# FONCTION DE TEST AMÉLIORÉE
async def test_router_connection():
    """Test complet de l'agent routeur et de sa capacité d'orchestration"""
    try:
        client = get_bedrock_client()
        if not client:
            return {
                "success": False,
                "error": "Client Bedrock indisponible",
                "solution": "Vérifiez les credentials AWS"
            }
        
        # Validation de la configuration
        config_check = validate_multi_agent_setup()
        if not config_check["valid"]:
            return {
                "success": False,
                "error": f"Configuration invalide: {', '.join(config_check['issues'])}",
                "solution": "Corrigez la configuration des agents"
            }
        
        session_id = get_or_create_session_id()
        
        # Test d'orchestration réel
        test_prompt = "Please orchestrate with your collaborator agents to analyze contract management best practices. This requires multi-agent collaboration."
        
        response = client.invoke_agent(
            agentId=AGENT_IDS["router"],
            agentAliasId=AGENT_ALIAS_IDS["router"],
            sessionId=session_id,
            inputText=test_prompt,
            enableTrace=True,
            endSession=False
        )
        
        parsed = parse_multi_agent_response_complete(response)
        
        # Évaluation des résultats
        collaboration_detected = bool(parsed["collaborator_responses"]) or any(
            "agent" in step.get("type", "") for step in parsed["orchestration_steps"]
        )
        
        return {
            "success": True,
            "response": parsed["final_response"][:300] + "..." if len(parsed["final_response"]) > 300 else parsed["final_response"],
            "collaboration_detected": collaboration_detected,
            "collaborators_count": len(parsed["collaborator_responses"]),
            "orchestration_steps": len(parsed["orchestration_steps"]),
            "note": "Test d'orchestration multi-agent réussi" if collaboration_detected else "Aucune collaboration détectée - vérifiez la configuration Bedrock"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:200],
            "solution": "Vérifiez la configuration de l'agent routeur dans AWS Bedrock"
        }

# PARSER MULTI-AGENT OPTIMISÉ
def parse_multi_agent_response_complete(response: Dict) -> Dict:
    """
    Parser optimisé pour les réponses multi-agent AWS Bedrock
    Gère correctement le streaming et l'orchestration
    """
    result = {
        "final_response": "",
        "collaborator_responses": {},
        "orchestration_steps": [],
        "trace_info": [],
        "errors": [],
        "raw_chunks": []  # Pour debug
    }
    
    try:
        # Traitement des événements de streaming
        event_stream = response.get("completion", [])
        
        for event in event_stream:
            # 1. CHUNKS - Réponse finale streamée
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    try:
                        decoded = chunk["bytes"].decode('utf-8')
                        result["raw_chunks"].append(decoded)
                        
                        # Filtrer les erreurs système
                        if any(error in decoded for error in ["RerunData", "InternalServerError", "ValidationException"]):
                            result["errors"].append(f"Erreur système filtrée: {decoded[:100]}")
                            continue
                        
                        # Essayer JSON puis texte brut
                        try:
                            chunk_json = json.loads(decoded)
                            if "text" in chunk_json:
                                result["final_response"] += chunk_json["text"]
                            elif "content" in chunk_json:
                                result["final_response"] += chunk_json["content"]
                        except json.JSONDecodeError:
                            if decoded.strip() and not decoded.startswith("{"):
                                result["final_response"] += decoded
                                
                    except UnicodeDecodeError as e:
                        result["errors"].append(f"Erreur décodage: {str(e)}")
            
            # 2. TRACES - Orchestration multi-agent
            elif "trace" in event:
                trace_event = event["trace"]
                
                # Informations du collaborateur
                collab_name = trace_event.get("collaboratorName")
                if collab_name:
                    result["trace_info"].append(f"Collaborateur: {collab_name}")
                
                # Contenu de la trace
                if "trace" in trace_event:
                    trace_data = trace_event["trace"]
                    
                    # ORCHESTRATION TRACE - Le plus important
                    if "orchestrationTrace" in trace_data:
                        orch = trace_data["orchestrationTrace"]
                        
                        # Raisonnement du routeur
                        if "modelInvocationInput" in orch:
                            reasoning = orch["modelInvocationInput"].get("text", "")
                            if reasoning:
                                result["orchestration_steps"].append({
                                    "type": "reasoning",
                                    "content": reasoning
                                })
                        
                        # Observations - Réponses des collaborateurs
                        if "observation" in orch:
                            obs = orch["observation"]
                            obs_type = obs.get("type")
                            
                            # AGENT COLLABORATOR - Réponse d'un agent
                            if obs_type == "AGENT_COLLABORATOR":
                                if "agentCollaboratorInvocationOutput" in obs:
                                    collab_out = obs["agentCollaboratorInvocationOutput"]
                                    agent_name = collab_out.get("agentCollaboratorName", "Agent")
                                    
                                    if "output" in collab_out and "text" in collab_out["output"]:
                                        agent_response = collab_out["output"]["text"]
                                        
                                        # Stocker la réponse
                                        result["collaborator_responses"][agent_name] = {
                                            "response": agent_response,
                                            "type": "collaborator"
                                        }
                                        
                                        result["orchestration_steps"].append({
                                            "type": "agent_response",
                                            "agent": agent_name,
                                            "preview": agent_response[:150] + "..." if len(agent_response) > 150 else agent_response
                                        })
                            
                            # FINISH - Réponse finale
                            elif obs_type == "FINISH":
                                if "finalResponse" in obs and "text" in obs["finalResponse"]:
                                    final_text = obs["finalResponse"]["text"]
                                    if final_text and len(final_text.strip()) > 0:
                                        # Priorité à la réponse finale de l'orchestration
                                        if not result["final_response"] or len(result["final_response"]) < len(final_text):
                                            result["final_response"] = final_text
                            
                            # ACTION GROUP
                            elif obs_type == "ACTION_GROUP":
                                if "actionGroupInvocationOutput" in obs:
                                    action_out = obs["actionGroupInvocationOutput"]
                                    action_text = action_out.get("text", "")
                                    result["orchestration_steps"].append({
                                        "type": "action",
                                        "content": action_text[:100] + "..." if len(action_text) > 100 else action_text
                                    })
                            
                            # KNOWLEDGE BASE
                            elif obs_type == "KNOWLEDGE_BASE":
                                if "knowledgeBaseLookupOutput" in obs:
                                    kb_out = obs["knowledgeBaseLookupOutput"]
                                    refs = kb_out.get("retrievedReferences", [])
                                    result["orchestration_steps"].append({
                                        "type": "knowledge_search",
                                        "references_count": len(refs)
                                    })
                    
                    # PRE/POST PROCESSING
                    for trace_type in ["preProcessingTrace", "postProcessingTrace"]:
                        if trace_type in trace_data:
                            trace_content = trace_data[trace_type]
                            if "modelInvocationInput" in trace_content:
                                input_text = trace_content["modelInvocationInput"].get("text", "")
                                if input_text:
                                    result["orchestration_steps"].append({
                                        "type": trace_type.replace("Trace", "").lower(),
                                        "content": input_text[:100] + "..." if len(input_text) > 100 else input_text
                                    })
    
    except Exception as e:
        result["errors"].append(f"Erreur parsing: {str(e)}")
        # En mode debug seulement
        if hasattr(st.session_state, 'debug_mode') and st.session_state.debug_mode:
            st.error(f"Erreur de parsing: {e}")
    
    # POST-TRAITEMENT
    # 1. Si pas de réponse finale, consolider les collaborateurs
    if not result["final_response"].strip() and result["collaborator_responses"]:
        sections = []
        for agent_name, agent_data in result["collaborator_responses"].items():
            # Trouver l'icône de l'agent
            agent_icon = "🤖"
            for key, info in AGENTS.items():
                if any(keyword in agent_name.lower() for keyword in [key, info["name"].lower().split()[0]]):
                    agent_icon = info["icon"]
                    break
            
            sections.append(f"{agent_icon} **{agent_name}**\n{agent_data['response']}")
        
        if sections:
            result["final_response"] = "\n\n".join(sections)
    
    # 2. Nettoyer la réponse finale
    if result["final_response"]:
        # Supprimer les doublons et nettoyer
        result["final_response"] = result["final_response"].strip()
        # Supprimer les répétitions de phrases
        lines = result["final_response"].split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
        result["final_response"] = '\n'.join(unique_lines)
    
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
            
            # Invoquer l'agent avec configuration optimisée
            invoke_params = {
                "agentId": AGENT_IDS[agent_key],
                "agentAliasId": AGENT_ALIAS_IDS[agent_key],
                "sessionId": current_session_id,
                "inputText": message_content,
                "enableTrace": True,
                "endSession": False
            }
            
            # Pour l'agent routeur, ajouter des paramètres spécifiques
            if agent_key == "router":
                # S'assurer que l'orchestration est activée
                invoke_params["enableTrace"] = True
                # Ajouter un contexte pour l'orchestration
                if "Orchestrate" not in message_content and "collaborate" not in message_content.lower():
                    invoke_params["inputText"] = f"Please orchestrate and collaborate with appropriate agents to handle this request: {message_content}"
            
            response = client.invoke_agent(**invoke_params)

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
            
            # Traitement spécial pour l'agent routeur
            if agent_key == "router":
                # Vérifier si l'orchestration a bien eu lieu
                has_collaboration = bool(parsed_response["collaborator_responses"]) or any(
                    step["type"] in ["agent_response", "collaborator_response"] 
                    for step in parsed_response["orchestration_steps"]
                )
                
                if has_collaboration:
                    # Orchestration réussie - formater la réponse
                    sections = []
                    
                    if parsed_response["final_response"]:
                        sections.append(f"🎯 **Orchestration Multi-Agent Complétée**\n\n{parsed_response['final_response']}")
                    
                    if parsed_response["collaborator_responses"]:
                        sections.append("\n---\n🤝 **Détails des Collaborateurs:**")
                        for agent_name, agent_data in parsed_response["collaborator_responses"].items():
                            agent_icon = "🤖"
                            for key, info in AGENTS.items():
                                if any(keyword in agent_name.lower() for keyword in [key, info["name"].lower().split()[0]]):
                                    agent_icon = info["icon"]
                                    break
                            sections.append(f"\n{agent_icon} **{agent_name}:**\n{agent_data['response']}")
                    
                    return "\n".join(sections)
                else:
                    # Pas de collaboration détectée - retourner la réponse directe
                    if parsed_response["final_response"]:
                        return f"🎯 **Agent Routeur (Réponse Directe):**\n\n{parsed_response['final_response']}"
                    else:
                        return f"⚠️ L'agent routeur n'a pas généré de réponse. Vérifiez la configuration multi-agent."
            else:
                # Autres agents - réponse standard
                return parsed_response["final_response"] if parsed_response["final_response"] else f"⚠️ Pas de réponse de {agent_name}"

        except Exception as e:
            error_str = str(e).lower()
            
            # Gestion spécifique des erreurs
            if "throttling" in error_str or "rate" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Backoff exponentiel
                    st.warning(f"⏳ Limite de débit. Attente {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return f"❌ Limite de débit dépassée pour {agent_name}. Réessayez plus tard."
            
            elif "accessdenied" in error_str or "forbidden" in error_str:
                return f"❌ Accès refusé pour {agent_name}. Vérifiez les permissions IAM et la configuration de l'agent."
            
            elif "resourcenotfound" in error_str or "notfound" in error_str:
                return f"❌ Agent {agent_name} introuvable. Vérifiez l'ID ({AGENT_IDS.get(agent_key, 'N/A')}) et l'alias ({AGENT_ALIAS_IDS.get(agent_key, 'N/A')})."
            
            elif "timeout" in error_str:
                if attempt < max_retries - 1:
                    st.warning(f"⏱️ Timeout pour {agent_name}. Nouvelle tentative...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    return f"❌ Timeout pour {agent_name}. L'orchestration peut prendre plus de temps que prévu."
            
            else:
                error_msg = f"❌ Erreur {agent_name}: {str(e)[:200]}"
                if attempt == max_retries - 1:  # Dernière tentative
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
    Workflow optimisé avec support multi-agent avancé
    """
    if mode == "intelligent":
        st.session_state.progress_text = f"🎯 Agent Routeur: Lancement de l'orchestration..."
        
        # Optimiser le prompt pour l'orchestration
        optimized_query = optimize_prompt_for_router(query)
        
        # Exécuter l'agent routeur avec le prompt optimisé
        response = await run_specific_agent(optimized_query, "router")
        
        # Enrichir la réponse avec des métadonnées
        response["selection_method"] = "Orchestration Multi-Agent Intelligente"
        response["original_query"] = query
        response["optimized_query"] = optimized_query
        response["mode"] = "intelligent_router"
        
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
    """Exécute une fonction asynchrone dans Streamlit avec gestion d'erreur améliorée"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))
    except Exception as e:
        st.error(f"Erreur d'exécution asynchrone: {str(e)}")
        return {"error": f"Erreur d'exécution: {str(e)}"}
    finally:
        try:
            loop.close()
        except:
            pass

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

# FONCTION D'OPTIMISATION DU PROMPT POUR ROUTEUR
def optimize_prompt_for_router(original_prompt: str) -> str:
    """Optimise le prompt pour déclencher l'orchestration multi-agent"""
    # Mots-clés qui déclenchent l'orchestration
    orchestration_triggers = [
        "analyze", "compare", "evaluate", "review", "draft", "negotiate", 
        "search", "quality", "management", "contract", "multiple", "comprehensive"
    ]
    
    # Vérifier si le prompt contient déjà des déclencheurs
    has_triggers = any(trigger in original_prompt.lower() for trigger in orchestration_triggers)
    
    if not has_triggers:
        # Ajouter un contexte d'orchestration
        return f"Please orchestrate with your collaborator agents to comprehensively address this request: {original_prompt}"
    
    return original_prompt

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

# FONCTION DE VALIDATION DE CONFIGURATION
def validate_multi_agent_setup():
    """Valide la configuration multi-agent complète"""
    issues = []
    
    # Vérifier les IDs d'agents
    for agent_key, agent_info in AGENTS.items():
        if agent_key not in AGENT_IDS or not AGENT_IDS[agent_key]:
            issues.append(f"ID manquant pour {agent_info['name']}")
        if agent_key not in AGENT_ALIAS_IDS or not AGENT_ALIAS_IDS[agent_key]:
            issues.append(f"Alias ID manquant pour {agent_info['name']}")
    
    # Vérifier la configuration spécifique du routeur
    if "router" in AGENT_IDS:
        router_id = AGENT_IDS["router"]
        if not router_id:
            issues.append("Agent routeur non configuré")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_agents": len(AGENTS),
        "configured_agents": len([k for k in AGENTS.keys() if k in AGENT_IDS and AGENT_IDS[k]])
    }
