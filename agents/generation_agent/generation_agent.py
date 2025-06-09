"""
Generation agent responsible for generating responses using an LLM.
"""
import os
import logging
from typing import Dict, Any, List
import google.generativeai as genai

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
        self.model_name = self.config.get("model", "gemini-1.5-flash")
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
                api_key_path = os.path.join(project_root, "tokenGemini.txt")
            
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
        Initialize the Gemini model.
        
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
                
            # Configure Gemini API with location settings
            location_settings = {
                "GOOGLE_APPLICATION_LOCATION_OVERRIDE": "us", # Ensure US region is used
            }
            
            # Set environment variables for location
            import os
            for key, value in location_settings.items():
                os.environ[key] = value
                
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # Get the model with minimal configuration
            # (full configuration will be provided during generation)
            self.model = genai.GenerativeModel(model_name=self.model_name)
            
            logger.info(f"Model {self.model_name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
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
        # Gemini-1.5-flash tiene un contexto total de 128K tokens
        # Reservamos ~6K tokens para la respuesta y estructura del prompt
        MAX_CONTEXT_TOKENS = 122000  
        
        # Estructura básica del prompt
        base_prompt = f"""Como asistente experto en bebidas y coctelería, responde a la siguiente consulta utilizando la información proporcionada.

Consulta: {query}

Información de referencia:
"""
        
        final_part = "\nBasándote en la información anterior, por favor responde a la consulta de manera concisa y profesional."
        
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
            
            # Generate response with safety settings for location
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            try:
                # Try with generation config and safety settings
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            except Exception as gen_error:
                logger.warning(f"First generation attempt failed: {gen_error}")
                try:
                    # Try without safety settings if there was an error
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback generation attempt failed: {fallback_error}")
                    return f"Lo siento, ocurrió un error al generar la respuesta con la API de Gemini. Por favor, verifica la configuración y los permisos de la API: {str(fallback_error)}"
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return "Lo siento, no pude generar una respuesta con la información disponible."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
