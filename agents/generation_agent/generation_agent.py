"""
Generation agent responsible for generating responses using an LLM.
"""
import os
import logging
from typing import Dict, Any, List
import requests
import json

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
        # Configuración para Fireworks AI
        self.model_name = self.config.get("model", "accounts/fireworks/models/mixtral-8x22b-instruct")
        self.temperature = self.config.get("temperature", 0.6)
        self.top_k = self.config.get("top_k", 32)
        self.top_p = self.config.get("top_p", 0.95)
        self.max_output_tokens = self.config.get("max_output_tokens", 512)
        
        # API key will be loaded from file at runtime
        self.api_key = None
        self.model = None
        
        # URL de la API de Fireworks (compatible con OpenAI)
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
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
        Initialize the Fireworks AI model connection.
        
        Args:
            api_key_path: Path to API key file
            
        Returns:
            Success flag
        """
        try:
            # Load API key if needed
            if not self.api_key and api_key_path:
                self.api_key = self._load_api_key(api_key_path)
            
            # Intenta buscar la clave API en la ubicación predeterminada si no se encontró
            if not self.api_key:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                default_paths = [
                    os.path.join(project_root, "tokenHuggingFace.txt"),
                    os.path.join(project_root, "tokenFireworks.txt"),
                    os.path.join(project_root, "tokenGemini.txt"),
                ]
                
                print(f"[Generation] Buscando API key en rutas predeterminadas...")
                for path in default_paths:
                    if os.path.exists(path):
                        print(f"[Generation] Intentando cargar clave desde {path}")
                        self.api_key = self._load_api_key(path)
                        if self.api_key:
                            print(f"[Generation] ✓ Clave API cargada correctamente desde {path}")
                            break
                
            if not self.api_key:
                print(f"[Generation] ❌ No se encontró ninguna clave API válida")
                logger.error("No API key available")
                return False
                
            # Para Fireworks AI, simplemente verificamos que la clave API esté disponible
            # La conexión real se hace en cada solicitud
            self.model = True
            
            logger.info(f"Model {self.model_name} connection initialized successfully using Fireworks AI API")
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
        
        # Estructura básica del prompt para el formato de chat
        base_prompt = f"""Como asistente experto en bebidas y coctelería, responde a la siguiente consulta utilizando la información proporcionada.

Consulta: {query}

Información de referencia:
"""
        
        final_part = "\nBasándote en la información anterior, por favor responde a la consulta de manera concisa y profesional. No incluyas referencias o menciones a las fuentes específicas en tu respuesta."
        
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
        logger.info(f"Tamaño del prompt: {len(prompt)} caracteres")
        
        # Limitar el tamaño final del prompt para evitar errores
        max_prompt_length = 8000  # Un límite conservador
        if len(prompt) > max_prompt_length:
            logger.warning(f"El prompt excede el tamaño máximo. Truncando de {len(prompt)} a {max_prompt_length} caracteres")
            prompt = prompt[:max_prompt_length]
        
        return prompt
    
    async def generate_response(self, query: str, search_results: List[Dict[str, Any]], api_key_path: str = None) -> str:
        """
        Generate a response using the LLM with Fireworks AI API.
        
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
            
            try:
                # Preparar la solicitud para la API de Fireworks
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                # Formatear el prompt como un mensaje de chat
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_output_tokens,
                    "stream": False
                }
                
                # Realizar la solicitud a la API con timeout
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()  # Lanzar excepción si hay error HTTP
                
                # Procesar la respuesta
                response_data = response.json()
                
                if "choices" not in response_data or not response_data["choices"]:
                    logger.error(f"Respuesta de API inesperada: {response_data}")
                    return "Lo siento, el servicio de IA no proporcionó una respuesta válida."
                
                # Extraer el texto generado
                generated_text = response_data["choices"][0]["message"]["content"]
                
                # Registrar información sobre la generación
                if "finish_reason" in response_data["choices"][0]:
                    finish_reason = response_data["choices"][0]["finish_reason"]
                    if finish_reason == "length":
                        logger.warning(f"La generación terminó por límite de longitud. Considera aumentar max_tokens.")
                    else:
                        logger.info(f"Generación completada. Razón: {finish_reason}")
                
                # Registrar información adicional útil para depuración
                logger.info(f"Respuesta generada con {len(generated_text)} caracteres")
                
                # Return generated text
                return generated_text
                
            except Exception as api_error:
                logger.error(f"API request error: {api_error}")
                error_details = ""
                if hasattr(api_error, 'response') and api_error.response:
                    try:
                        error_details = f" - Detalles: {api_error.response.json()}"
                    except:
                        error_details = f" - Código de estado: {api_error.response.status_code}"
                
                return f"Lo siento, ocurrió un error al generar la respuesta con la API de Fireworks AI: {str(api_error)}{error_details}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
