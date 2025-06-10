"""
Generation agent responsible for generating responses using an LLM.
"""
import os
import logging
from typing import Dict, Any, List
import requests
import json
from huggingface_hub import InferenceClient

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class GenerationAgent(Agent):
    """Agent responsible for LLM-based text generation"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the generation agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("generation_agent")
        self.data_store = DataStore()
        self.model_name = self.config.get("model", "mixtral-8x7b")
        self.temperature = self.config.get("temperature", 0.4)
        self.top_k = self.config.get("top_k", 32)
        self.top_p = self.config.get("top_p", 0.95)
        self.max_output_tokens = self.config.get("max_output_tokens", 2048)
        
        # API key will be loaded from file at runtime
        self.api_key = None
        self.model = None
        
    async def start(self) -> None:
        """Start the generation agent"""
        await super().start()
        logger.info(f"Generation agent {self.agent_id} started")
    
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
        
        if action == "generate_response":
            query = content.get("query", "")
            search_results = content.get("search_results", [])
            api_key_path = content.get("api_key_path")
            
            # Validate inputs
            if not query:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "generation_results",
                        "status": "error",
                        "message": "Empty query"
                    }
                }
                
            if not api_key_path and not self.api_key:
                # Look for token file in default location
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                api_key_path = os.path.join(project_root, "tokenHuggingFace.txt")
            
            # Generate response
            response = await self.generate_response(query, search_results, api_key_path)
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "generation_results",
                    "status": "success" if response else "error",
                    "query": query,
                    "response": response,
                    "references": len(search_results)
                }
            }
            
        return None
    
    def _load_api_key(self, api_key_path: str) -> str:
        """
        Load API key from file.
        
        Args:
            api_key_path: Path to API key file
            
        Returns:
            API key as string
        """
        try:
            with open(api_key_path, 'r') as f:
                content = f.read().strip()
                
                # Remove any comments
                if '//' in content:
                    content = content[:content.find('//')].strip()
                
                return content
                
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            return None
    
    def _initialize_model(self, api_key_path: str = None) -> bool:
        """
        Initialize the Mixtral model connection.
        
        Args:
            api_key_path: Path to API key file
            
        Returns:
            Success flag
        """
        try:
            # Load API key if needed
            if not self.api_key and api_key_path:
                self.api_key = self._load_api_key(api_key_path)
                
            if not self.api_key:
                logger.error("No API key available")
                return False
                
            # For Mixtral, we don't need to initialize a model object here
            # We'll just verify the API key is available
            self.model = True
            
            logger.info(f"Model {self.model_name} connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model connection: {e}")
            return False
    
    def _build_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Build a prompt with query and search results context.
        Includes as many documents as possible within token limits.
        
        Args:
            query: User query
            search_results: Search results to include as context
            
        Returns:
            Formatted prompt
        """
        # Estimar el tamaño promedio de token para controlar el tamaño del prompt
        # Un estimado conservador es aproximadamente 4 caracteres por token
        CHARS_PER_TOKEN = 4
        
        # Estimar tokens máximos para el modelo (para dejar espacio para la respuesta)
        # Mixtral-8x7b tiene un contexto total de aproximadamente 32K tokens
        # Reservamos ~2K tokens para la respuesta y estructura del prompt
        MAX_CONTEXT_TOKENS = 30000  
        
        # Estructura básica del prompt con formato específico para Mixtral Instruct
        base_prompt = f"""<s>[INST] Como asistente experto en bebidas y coctelería, responde a la siguiente consulta utilizando la información proporcionada.

Consulta: {query}

Información de referencia:
"""
        
        final_part = "\nBasándote en la información anterior, por favor responde a la consulta de manera concisa y profesional. No incluyas referencias o menciones a las fuentes específicas en tu respuesta. [/INST]"
        
        # Estimar tokens ya utilizados
        base_tokens = (len(base_prompt) + len(final_part)) // CHARS_PER_TOKEN
        available_tokens = MAX_CONTEXT_TOKENS - base_tokens
        
        # Ordenar resultados por score (si está disponible)
        if search_results and "score" in search_results[0]:
            search_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)
        
        # Construir el prompt dinámicamente, añadiendo tantos documentos como quepan
        used_sources = []
        current_tokens = 0
        
        for i, result in enumerate(search_results, 1):
            title = result.get("title", "Sin título")
            snippet = result.get("snippet", "")
            
            # Crear el fragmento para este resultado
            source_text = f"\nFuente {i}: {title}\n{snippet}\n"
            source_tokens = len(source_text) // CHARS_PER_TOKEN
            
            # Verificar si aún hay espacio para este documento
            if current_tokens + source_tokens <= available_tokens:
                used_sources.append(source_text)
                current_tokens += source_tokens
            else:
                # No hay más espacio, detener la adición de fuentes
                break
        
        # Construir el prompt final
        prompt = base_prompt
        
        # Añadir las fuentes que pudieron ser incluidas
        for source in used_sources:
            prompt += source
            
        # Añadir la parte final del prompt
        prompt += final_part
        
        logger.info(f"Prompt construido con {len(used_sources)} de {len(search_results)} documentos disponibles")
        
        return prompt
    
    async def generate_response(self, query: str, search_results: List[Dict[str, Any]], api_key_path: str = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query: User query
            search_results: Search results for context
            api_key_path: Path to API key file
            
        Returns:
            Generated response
        """
        try:
            logger.info(f"Generating response for query: {query}")
            
            # Initialize model if needed
            if not self.model:
                if not self._initialize_model(api_key_path):
                    return "Lo siento, no puedo generar una respuesta en este momento debido a problemas con la configuración del modelo."
            
            # Build the prompt
            prompt = self._build_prompt(query, search_results)
            
            # Initialize the Hugging Face Inference Client
            client = InferenceClient(token=self.api_key)
            
            try:
                # Generate text using Hugging Face Inference API
                # Usamos un valor mucho más alto para max_new_tokens para evitar límites en la respuesta
                response = client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=4096,  # Valor aumentado para permitir respuestas más largas
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                    details=True,  # Obtener detalles adicionales sobre la generación
                    return_full_text=False  # Solo queremos el texto generado, no el prompt
                )
                
                # Extraer el texto generado y registrar información de la generación
                generated_text = response.generated_text
                
                # Registrar información sobre la generación (si está disponible)
                if hasattr(response, 'details') and response.details:
                    finish_reason = response.details.finish_reason
                    if finish_reason == 'length':
                        logger.warning(f"La generación terminó por límite de longitud. Considera aumentar max_new_tokens.")
                    else:
                        logger.info(f"Generación completada. Razón: {finish_reason}")
                
                # Return generated text
                return generated_text
                
            except Exception as api_error:
                logger.error(f"API request error: {api_error}")
                return f"Lo siento, ocurrió un error al generar la respuesta con la API de Hugging Face: {str(api_error)}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
