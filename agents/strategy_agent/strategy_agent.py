"""
Strategy agent responsible for choosing between embeddings and ontology approaches.
"""
import os
import logging
from typing import Dict, Any, List, Tuple, Union
import json
import asyncio

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
        self.model_name = self.config.get("model", "mistral-medium")  # Modelos disponibles: mistral-tiny, mistral-small, mistral-medium
        self.temperature = self.config.get("temperature", 0.1)  # Low temperature for more deterministic responses
        self.api_key = None
        self.client = None
        
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
        """Load API key from file for Mistral AI API"""
        try:
            # Cargar la API key
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            token_file = os.path.join(project_root, "tokenHuggingFace.txt")
            with open(token_file, 'r') as f:
                self.api_key = f.read().strip()
                logger.info("API key loaded successfully")
                
            # Configurar el cliente como activo (usaremos requests directamente)
            self.client = True  # Marcamos como inicializado
            logger.info("API connection ready for Mistral AI")
                
        except Exception as e:
            logger.error(f"Failed to load API key: {e}")
            self.api_key = None
            self.client = None
    
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
            strategy, needs_dynamic_crawling, explanation = await self._determine_strategy(query)
            
            sender = message.get("sender") or "coordinator_agent"  # Si no hay 'sender', usar coordinator_agent
            logger.info(f"[StrategyAgent] Preparando respuesta para enviar a {sender} (op_id: {operation_id})")
            
            # Si se requiere información dinámica, activar el crawler dinámico inmediatamente
            if needs_dynamic_crawling:
                logger.info(f"[StrategyAgent] Activando crawler dinámico para '{query}' (op_id: {operation_id})")
                try:
                    # Enviar solicitud al crawler dinámico primero
                    await self.send_message("dynamic_crawler_agent", {
                        "action": "search_web",
                        "query": query,
                        "operation_id": operation_id,
                        "source": "strategy",
                        "callback_agent": "coordinator_agent"  # Siempre enviar resultados al coordinador
                    })
                    logger.info(f"[StrategyAgent] Mensaje enviado correctamente al crawler dinámico")
                except Exception as e:
                    logger.error(f"[StrategyAgent] Error al enviar mensaje al crawler dinámico: {e}")
            
            response = {
                "recipient": sender,
                "content": {
                    "action": "strategy_response",
                    "operation_id": operation_id,
                    "strategy": strategy,
                    "skip_embedding_search": needs_dynamic_crawling,  # Indica al coordinador si debe omitir la búsqueda por embeddings
                    "explanation": explanation,
                    "needs_dynamic_crawling": needs_dynamic_crawling,
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
    
    async def _determine_strategy(self, query: str) -> Tuple[str, bool]:
        """
        Determine the best strategy to use for a given query.
        
        Args:
            query: The user's search query
        
        Returns:
            A tuple of (strategy, needs_dynamic_crawling) where strategy is either "embedding" or "ontology"
            and needs_dynamic_crawling is a boolean indicating if dynamic web crawling is needed
        """
        # Check if we have a cached decision for this query
        if query in self.decision_cache:
            return self.decision_cache[query]
        
        if not self.api_key or not self.client:
            logger.warning("No API key or Mistral client available, falling back to embeddings approach")
            logger.error(f"[StrategyAgent] ERROR: API key o cliente Mistral no disponible. ¿Se llamó al método start() para cargarla?")
            return "embedding", False
        
        try:
            # Construct prompt for strategy decision with dynamic crawling evaluation
            system_prompt = """
            Tu tarea es determinar la mejor estrategia para responder a una consulta sobre cócteles.

            Sigue este procedimiento paso a paso:

            ### Paso 1: Analiza la intención de la consulta
            Clasifica la consulta según su tipo:

            - (1) Estructurada: definición breve, clasificación, taxonomía, relaciones explícitas o elementos concretos (ingredientes, utensilios, categorías). 
              Ej.: “¿Qué ingredientes lleva el Negroni?”, “¿Qué categoría es el Bloody Mary?”
              ⚠️ No consideres como estructurada una consulta que busca historia o explicaciones narrativas, aunque sea concreta.

            - (2) Semántica general: recomendaciones, lenguaje subjetivo, comparaciones o conceptos similares. 
              Ej.: “¿Qué cócteles son parecidos al Mojito?”, “Cócteles para una cita romántica”

            - (3) Temporal o dinámica: información reciente, tendencias, eventos actuales, novedades o que varía en el tiempo. 
              Ej.: “Cócteles populares en 2024”, “Tendencias actuales en mixología”

            ### Paso 2: Selecciona el método de búsqueda

            - Usa `"ontology"` solo si la pregunta es estructurada (tipo 1), y requiere información factual, categorizada o relacional. 
              No uses `"ontology"` para temas narrativos o explicativos como historia, aunque sean consultas concretas.

            - Usa `"embedding"` si la consulta es semántica (tipo 2), subjetiva, ambigua, en lenguaje libre, busca explicaciones narrativas o términos similares.
              Ejemplos:
              - “¿Qué hace único al Paper Plane?”
              - “Historia de la Coca-Cola”
              - “Cócteles como el Mojito”

            Si hay ambigüedad entre estructurada y semántica, **elige embedding** como alternativa segura.

            ### Paso 3: Decide si se necesita información de la web en tiempo real (`crawl`)

            - Usa `"crawl": true` **solo si**:
              - La consulta menciona años, décadas, siglos o términos como: “nuevos”, “recientes”, “actualmente”, “últimas tendencias”, “creados en…”
              - Se trata de temas que cambian en el tiempo o no suelen estar en una base de conocimiento estática (por ejemplo, rankings, lanzamientos, modas)

              Ejemplos:
              - “¿Cuáles son las últimas tendencias en cócteles?”
              - “Cócteles populares en 2023”
              - “Nuevos cócteles con gin en 2024”

            - Usa `"crawl": false` si:
              - La consulta trata sobre historia general, preparación, categorías o conocimientos que pueden estar almacenados en la base local (aunque históricos)
              - La información es estática o ampliamente conocida

             **No actives el crawler solo por tratarse de una pregunta histórica si no se requiere actualización reciente**

            ### Responde únicamente en formato JSON:

            {
              "strategy": "ontology" | "embedding",
              "crawl": true | false,
              "explanation": "Máximo 2 líneas explicando tu decisión"
            }
            """

            # Usar un formato de mensaje estándar para APIs de chat
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Consulta del usuario: {query}"}
            ]
            
            # Configurar la solicitud para la API de Mistral
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # URL de la API de Mistral
            api_url = "https://api.mistral.ai/v1/chat/completions"
            
            # Preparar el payload según la documentación de Mistral AI
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": 0.95,
                "max_tokens": 150,
                "safe_prompt": True
            }
            
            logger.info(f"Enviando petición a Mistral AI para determinar estrategia para: '{query}'")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            # Verificar la respuesta
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"Respuesta de Mistral AI: {content}")
            
            # Intentar analizar la respuesta JSON
            try:
                # Limpiar la respuesta de marcadores de formato
                cleaned_content = content
                
                # Eliminar marcadores de código comunes
                for marker in ["```json", "```"]:
                    cleaned_content = cleaned_content.replace(marker, "")
                
                cleaned_content = cleaned_content.strip()
                
                decision = json.loads(cleaned_content)
                strategy = decision.get("strategy", "embedding").lower()
                needs_crawling = decision.get("crawl", False)
                explanation = decision.get("explanation", "No explanation provided")
                
                # Validar strategy
                if strategy not in ["ontology", "embedding"]:
                    logger.warning(f"Invalid strategy '{strategy}' from LLM, defaulting to embedding")
                    strategy = "embedding"
                
                logger.info(f"[StrategyAgent] Determinada estrategia: {strategy} (crawling: {needs_crawling}) para la consulta '{query}'")
                logger.info(f"[StrategyAgent] Explicación: {explanation}")
                
                # Cachear la decisión
                self.decision_cache[query] = (strategy, needs_crawling, explanation)
                
                return strategy, needs_crawling, explanation
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON, defaulting to embedding approach without crawling")
                # Intentar extraer la estrategia y decisión de crawling del texto
                content_lower = content.lower()
                
                strategy = "embedding"
                if "ontology" in content_lower:
                    strategy = "ontology"
                
                needs_crawling = "crawl" in content_lower and "true" in content_lower
                
                explanation = "Extracted from non-JSON response"
                self.decision_cache[query] = (strategy, needs_crawling, explanation)
                
                return strategy, needs_crawling, explanation
                
        except Exception as e:
            logger.error(f"Error determining strategy: {e}")
            return "embedding", False, "Error determining strategy"
    
    async def fallback_strategy(self) -> Tuple[str, str]:
        """
        Return a fallback strategy when the LLM-based decision fails.
        
        Returns:
            A tuple of (strategy, explanation) with the default fallback strategy
        """
        return "embedding", "Fallback to embedding approach"
    
    async def determine_strategy(self, query: str) -> Tuple[str, bool, str]:
        """
        Public method to determine the best strategy for answering the query.
        
        Args:
            query: The user's query
            
        Returns:
            A tuple of (strategy, needs_dynamic_crawling, explanation)
        """
        try:
            return await self._determine_strategy(query)
        except Exception as e:
            logger.error(f"Error in determine_strategy: {e}")
            return "embedding", False, "Error determining strategy, using embedding as fallback"
