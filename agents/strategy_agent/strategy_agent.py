"""
Strategy agent responsible for choosing between embeddings and ontology approaches.
"""
import os
import logging
from typing import Dict, Any, List, Tuple
import json
import requests

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class StrategyAgent(Agent):
    """Agent responsible for choosing the optimal search strategy"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the strategy agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("strategy_agent")
        self.data_store = DataStore()
        
        # LLM configuration
        self.model_name = self.config.get("model", "accounts/fireworks/models/mixtral-8x22b-instruct")
        self.temperature = self.config.get("temperature", 0.1)  # Low temperature for more deterministic responses
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.api_key = None
        
        # Cache de decisiones para consultas similares
        self.decision_cache = {}
        
    async def start(self) -> None:
        """Start the strategy agent"""
        await super().start()
        
        # Cargar API key
        self._load_api_key()
        
        if self.api_key:
            logger.info(f"Strategy agent {self.agent_id} started with valid API key")
        else:
            logger.error(f"Strategy agent {self.agent_id} started but failed to load API key")
    
    def _load_api_key(self):
        """Load API key from file"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            token_file = os.path.join(project_root, "tokenHuggingFace.txt")
            with open(token_file, 'r') as f:
                self.api_key = f.read().strip()
                logger.info("API key loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load API key: {e}")
            self.api_key = None
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        content = message.get("content", {})
        action = content.get("action")
        
        if action == "determine_strategy":
            query = content.get("query", "")
            operation_id = content.get("operation_id", "unknown")
            
            # Process the query to determine the best strategy
            logger.info(f"[StrategyAgent] Determinando estrategia para '{query}' (op_id: {operation_id})")
            strategy, explanation = await self._determine_strategy(query)
            
            sender = message.get("sender") or "coordinator_agent"  # Si no hay 'sender', usar coordinator_agent
            logger.info(f"[StrategyAgent] Preparando respuesta para enviar a {sender} (op_id: {operation_id})")
            response = {
                "recipient": sender,
                "content": {
                    "action": "strategy_response",
                    "operation_id": operation_id,
                    "strategy": strategy,
                    "explanation": explanation,
                    "query": query
                }
            }
            logger.info(f"[StrategyAgent] Respuesta preparada: {response}")
            return response
        else:
            logger.warning(f"Unknown action '{action}' received by strategy agent")
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "error",
                    "error": f"Unknown action: {action}",
                    "operation_id": content.get("operation_id", "unknown")
                }
            }
    
    async def _determine_strategy(self, query: str) -> Tuple[str, str]:
        """
        Determine the best strategy to use for a given query.
        
        Args:
            query: The user's search query
            
        Returns:
            A tuple of (strategy, explanation) where strategy is either "embedding" or "ontology"
        """
        # Check if we have a cached decision for this query
        if query in self.decision_cache:
            return self.decision_cache[query]
        
        if not self.api_key:
            logger.warning("No API key available, falling back to embeddings approach")
            logger.error(f"[StrategyAgent] ERROR: API key no disponible. ¿Se llamó al método start() para cargarla?")
            return "embedding", "API key not available for LLM decision"
        
        try:
            # Construct prompt for strategy decision
            system_prompt = """
            Tu tarea es determinar qué enfoque de búsqueda es más adecuado para una consulta dada:
            
            1. Búsqueda basada en ONTOLOGÍA: Ideal para consultas estructuradas, definiciones, 
               relaciones explícitas, clasificaciones y cuando se necesita precisión en dominios específicos.
               Ejemplos: "¿Qué es un Manhattan?", "¿A qué categoría pertenece el Martini?", 
               "¿Qué cócteles usan ginebra como base?"
               
            2. Búsqueda basada en EMBEDDINGS: Ideal para similitud semántica, consultas en lenguaje natural, 
               recomendaciones, y cuando se buscan conceptos relacionados sin una estructura formal.
               Ejemplos: "¿Cómo preparar un cóctel refrescante?", "Cócteles similares al Mojito", 
               "Bebidas para una fiesta de verano"
            
            Responde en formato JSON con dos campos:
            - "strategy": "ontology" o "embedding"
            - "explanation": breve explicación de tu decisión (máximo 2 líneas)
            """
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Consulta del usuario: {query}"}
                ],
                "temperature": self.temperature,
                "top_p": 0.95,
                "max_tokens": 150
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            logger.info(f"Sending strategy determination request to LLM for query: '{query}'")
            logger.info(f"[StrategyAgent] Enviando petición a API URL: {self.api_url}")
            logger.info(f"[StrategyAgent] Auth header: Bearer {self.api_key[:4]}...")
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)  # Timeout de 30 segundos
            status_code = response.status_code
            logger.info(f"[StrategyAgent] API response status code: {status_code}")
            
            if status_code != 200:
                logger.error(f"[StrategyAgent] Error en la respuesta: {response.text[:200]}...")
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"[StrategyAgent] LLM response (resumido): {str(result)[:200]}...")
            logger.debug(f"LLM response: {result}")
            
            # Extract the assistant's message content
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            try:
                # Parse JSON response
                decision = json.loads(content)
                strategy = decision.get("strategy", "embedding").lower()
                explanation = decision.get("explanation", "No explanation provided")
                
                # Validate strategy
                if strategy not in ["ontology", "embedding"]:
                    logger.warning(f"Invalid strategy '{strategy}' from LLM, defaulting to embedding")
                    strategy = "embedding"
                    explanation = "Default to embedding due to invalid LLM response"
            except Exception as e:
                logger.warning(f"Failed to parse LLM response: {e}, defaulting to embedding approach")
                strategy = "embedding"
                explanation = "Default to embedding due to LLM response parsing error"
            
            # Cache the decision
            self.decision_cache[query] = (strategy, explanation)
            
            logger.info(f"[StrategyAgent] Determinada estrategia: {strategy} para la consulta '{query}'")
            logger.info(f"[StrategyAgent] Explicación: {explanation}")
            
            return strategy, explanation
        
        except Exception as e:
            logger.error(f"Error determining strategy: {e}")
            return "embedding", f"Error determining strategy, defaulting to embedding approach"
    
    async def fallback_strategy(self) -> Tuple[str, str]:
        """
        Return a fallback strategy when the LLM-based decision fails.
        
        Returns:
            A tuple of (strategy, explanation) with the default fallback strategy
        """
        return "embedding", "Fallback to embedding approach"
