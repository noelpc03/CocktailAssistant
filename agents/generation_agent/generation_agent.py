"""
Generation agent responsible for generating responses using an LLM.
"""
import os
import logging
from typing import Dict, Any, List
import json
import asyncio

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
        # Configuración para Mistral AI
        self.model_name = self.config.get("model", "mistral-medium")  # Modelos disponibles: mistral-tiny, mistral-small, mistral-medium
        self.temperature = self.config.get("temperature", 0.6)
        self.top_p = self.config.get("top_p", 0.95)
        self.max_output_tokens = self.config.get("max_output_tokens", 512)
        
        # API key will be loaded from file at runtime
        self.api_key = None
        self.client = None
        self.model = None  # Indicador de si el modelo está inicializado
        
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
            
            # Comprobar si hay resultados de ontología (formato nuevo)
            ontology_results = content.get("ontology_results", [])
            
            # Comprobar si hay información web del crawler dinámico
            web_info = content.get("web_info", "")
            
            api_key_path = content.get("api_key_path")
            operation_id = content.get("operation_id", "")
            # Check if dynamic crawling is needed (propagated from coordinator)
            needs_dynamic_crawling = content.get("needs_dynamic_crawling", False)
            ontology_failed = content.get("ontology_failed", False)
            
            # Validate inputs
            if not query:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "generation_results",
                        "status": "error",
                        "message": "Empty query",
                        "operation_id": operation_id,
                        "needs_dynamic_crawling": needs_dynamic_crawling
                    }
                }
                
            if not api_key_path and not self.api_key:
                # Look for token file in default location
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                api_key_path = os.path.join(project_root, "tokenHuggingFace.txt")
            
            # Determine the best strategy for generating a response
            # Prioridad: 1) Web + Ontología, 2) Web + Embeddings, 3) Web sola, 4) Ontología, 5) Embeddings
            
            # Para consultas temporales/dinámicas, priorizar información web siempre
            if web_info:
                logger.info("Web information available, prioritizing it in response generation")
                if ontology_results:
                    # Tenemos tanto resultados de ontología como información web
                    print(f"[Generation] Generando respuesta combinando ontología y web")
                    response = await self.generate_response_from_combined_sources(query, ontology_results, web_info, api_key_path)
                elif search_results and len(search_results) > 0:
                    # Tenemos tanto resultados de embeddings como información web
                    print(f"[Generation] Generando respuesta combinando embeddings y web")
                    response = await self.generate_response_from_web_and_embeddings(query, search_results, web_info, api_key_path)
                else:
                    # Solo tenemos información web
                    print(f"[Generation] Generando respuesta a partir de información web")
                    response = await self.generate_response_from_web_info(query, web_info, api_key_path, ontology_failed)
            elif ontology_results:
                # Solo tenemos resultados de ontología
                print(f"[Generation] Generando respuesta a partir de resultados de ontología")
                response = await self.generate_response_from_ontology(query, ontology_results, api_key_path)
            else:
                # Proceso normal para búsquedas regulares por embeddings
                print(f"[Generation] Generando respuesta a partir de búsqueda por embeddings")
                response = await self.generate_response(query, search_results, api_key_path)
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "generation_results",
                    "status": "success" if response else "error",
                    "query": query,
                    "response": response,
                    "references": len(search_results),
                    "operation_id": operation_id,
                    "needs_dynamic_crawling": needs_dynamic_crawling
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
        Initialize the Mistral AI model connection.
        
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
                
            # Para la API de Mistral, simplemente verificamos que la clave API esté disponible
            # La conexión real se hace en cada solicitud
            self.client = True  # Marcamos como inicializado
            self.model = True
            
            logger.info(f"Model {self.model_name} connection initialized successfully using Mistral AI API")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model connection: {e}")
            return False
            
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
        Generate a response using the LLM with Mistral AI API.
        
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
                import requests
                
                # Crear el mensaje para Mistral AI
                messages = [{"role": "user", "content": prompt}]
                
                # Configurar la solicitud para la API de Mistral
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
                    "top_p": self.top_p,
                    "max_tokens": self.max_output_tokens,
                    "safe_prompt": True
                }
                
                # Realizar la solicitud a la API con timeout
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()  # Lanzar excepción si hay error HTTP
                
                # Convertir la respuesta a JSON
                response_data = response.json()
                
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
                
            except ImportError as import_error:
                logger.error(f"Error importing required dependencies: {import_error}")
                return f"Lo siento, ocurrió un error con las dependencias requeridas: {str(import_error)}"
                
            except Exception as api_error:
                logger.error(f"API request error: {api_error}")
                return f"Lo siento, ocurrió un error al generar la respuesta con la API de Mistral AI: {str(api_error)}"
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
        
    async def generate_response_from_ontology(self, query: str, ontology_results: List[Dict[str, Any]], api_key_path: str = None) -> str:
        """
        Genera una respuesta usando LLM a partir de resultados de consulta de ontología.
        Este método reemplaza el proceso doble de generación y permite usar LLM una sola vez.
        
        Args:
            query: Consulta del usuario
            ontology_results: Resultados de la consulta SPARQL a la ontología
            api_key_path: Ruta al archivo de clave API
            
        Returns:
            Respuesta generada
        """
        try:
            logger.info(f"Generando respuesta para consulta de ontología: {query}")
            
            # Inicializar modelo si es necesario
            if not self.model:
                if not self._initialize_model(api_key_path):
                    return "Lo siento, no puedo generar una respuesta en este momento debido a problemas con la configuración del modelo."
            
            # Formatear los resultados de la ontología para el prompt
            results_text = json.dumps(ontology_results, indent=2, ensure_ascii=False)
            
            # Construir prompt especial para resultados de ontología
            prompt = f"""Como experto en cócteles, responde a la siguiente consulta utilizando SOLO los resultados de la ontología proporcionados.

Consulta: {query}

Resultados de la ontología (consulta SPARQL):
{results_text}

Tu respuesta debe:
1. Ser concisa pero informativa
2. Incluir cantidades específicas para los ingredientes cuando corresponda (ej: 60ml de gin, 15ml de vermú)
3. Incluir instrucciones básicas de preparación si corresponde
4. Tener un tono conversacional y profesional 
5. No mencionar detalles técnicos como SPARQL, ontologías, o URIs

Responde directamente a la consulta con toda la información relevante de los resultados.
"""
            
            try:
                import requests
                
                # Crear el mensaje para Mistral AI
                messages = [{"role": "user", "content": prompt}]
                
                # Configurar la solicitud para la API de Mistral
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
                    "top_p": self.top_p,
                    "max_tokens": self.max_output_tokens,
                    "safe_prompt": True
                }
                
                # Realizar la solicitud a la API con timeout
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()  # Lanzar excepción si hay error HTTP
                
                # Convertir la respuesta a JSON
                response_data = response.json()
                
                # Extraer el texto generado
                generated_text = response_data["choices"][0]["message"]["content"]
                
                # Registrar información sobre la generación
                logger.info(f"Respuesta de ontología generada con {len(generated_text)} caracteres")
                print(f"[Generation] ✓ Respuesta generada exitosamente (una única llamada a LLM)")
                
                # Return generated text
                return generated_text
                
            except ImportError as import_error:
                logger.error(f"Error importing required dependencies: {import_error}")
                return f"Lo siento, ocurrió un error con las dependencias requeridas: {str(import_error)}"
                
            except Exception as api_error:
                logger.error(f"API request error: {api_error}")
                return f"Lo siento, ocurrió un error al generar la respuesta con la API de Mistral AI: {str(api_error)}"
                
        except Exception as e:
            logger.error(f"Error generating response from ontology results: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta a partir de los resultados de la ontología: {str(e)}"
    
    async def generate_response_from_web_info(self, query: str, web_info: str, api_key_path: str = None, ontology_failed: bool = False) -> str:
        """
        Genera una respuesta usando LLM a partir de la información obtenida de la web.
        
        Args:
            query: Consulta del usuario
            web_info: Información extraída de búsquedas web
            api_key_path: Ruta al archivo de clave API
            ontology_failed: Indica si la consulta de ontología falló
            
        Returns:
            Respuesta generada
        """
        try:
            logger.info(f"Generando respuesta para consulta con info web: {query}")
            
            # Inicializar modelo si es necesario
            if not self.model:
                if not self._initialize_model(api_key_path):
                    return "Lo siento, no puedo generar una respuesta en este momento debido a problemas con la configuración del modelo."
            
            # Construir prompt especial para información web
            context_note = "que no se encontró en nuestra base de conocimientos" if ontology_failed else "complementaria"
            
            prompt = f"""Como experto en cócteles, responde a la siguiente consulta utilizando la información web {context_note}.

Consulta: {query}

Información Web:
{web_info}

Tu respuesta debe:
1. Ser concisa pero informativa
2. Incluir información precisa sobre los cócteles, ingredientes y preparación cuando corresponda
3. Tener un tono conversacional y profesional
4. No mencionar que la información proviene de búsquedas web o mencionar fuentes específicas

Si algún detalle no está claro en la información proporcionada, indícalo de manera sutil.
"""
            
            try:
                import requests
                
                # Crear el mensaje para Mistral AI
                messages = [{"role": "user", "content": prompt}]
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_output_tokens
                    }
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"Error in API response: {response.status_code} - {response.text}")
                    return f"Lo siento, ocurrió un error al consultar el modelo de lenguaje (código {response.status_code})."
                    
            except Exception as e:
                logger.error(f"Error generating response from web info: {e}")
                return f"Lo siento, ocurrió un error al procesar la información de la web: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in generate_response_from_web_info: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
    
    async def generate_response_from_combined_sources(self, query: str, ontology_results: List[Dict[str, Any]], web_info: str, api_key_path: str = None) -> str:
        """
        Genera una respuesta usando LLM combinando resultados de la ontología e información web.
        
        Args:
            query: Consulta del usuario
            ontology_results: Resultados de la consulta SPARQL a la ontología
            web_info: Información extraída de búsquedas web
            api_key_path: Ruta al archivo de clave API
            
        Returns:
            Respuesta generada
        """
        try:
            logger.info(f"Generando respuesta combinada (ontología + web) para consulta: {query}")
            
            # Inicializar modelo si es necesario
            if not self.model:
                if not self._initialize_model(api_key_path):
                    return "Lo siento, no puedo generar una respuesta en este momento debido a problemas con la configuración del modelo."
            
            # Formatear los resultados de la ontología para el prompt
            results_text = json.dumps(ontology_results, indent=2, ensure_ascii=False)
            
            # Construir prompt para combinar fuentes
            prompt = f"""Como experto en cócteles, responde a la siguiente consulta utilizando TANTO la información estructurada de nuestra base de conocimientos COMO la información web complementaria.

Consulta: {query}

Resultados de la base de conocimientos (consulta SPARQL):
{results_text}

Información Web Complementaria:
{web_info}

Tu respuesta debe:
1. Priorizar la información estructurada de la base de conocimientos cuando sea relevante y precisa
2. Complementar con la información web donde sea necesario para dar respuestas más completas
3. Resolver cualquier contradicción favoreciendo la fuente más confiable
4. Ser concisa pero informativa
5. Tener un tono conversacional y profesional
6. No mencionar detalles técnicos como SPARQL, ontologías, o URIs, ni mencionar que usas múltiples fuentes

Responde directamente a la consulta combinando toda la información disponible de manera coherente.
"""
            
            try:
                import requests
                
                # Crear el mensaje para Mistral AI
                messages = [{"role": "user", "content": prompt}]
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_output_tokens
                    }
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"Error in API response: {response.status_code} - {response.text}")
                    return f"Lo siento, ocurrió un error al consultar el modelo de lenguaje (código {response.status_code})."
                    
            except Exception as e:
                logger.error(f"Error generating combined response: {e}")
                return f"Lo siento, ocurrió un error al procesar la información combinada: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in generate_response_from_combined_sources: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
    
    async def generate_response_from_web_and_embeddings(self, query: str, search_results: List[Dict[str, Any]], web_info: str, api_key_path: str = None) -> str:
        """
        Genera una respuesta usando LLM combinando resultados de embeddings e información web.
        
        Args:
            query: Consulta del usuario
            search_results: Resultados de la búsqueda por embeddings
            web_info: Información extraída de búsquedas web
            api_key_path: Ruta al archivo de clave API
            
        Returns:
            Respuesta generada
        """
        try:
            logger.info(f"Generando respuesta combinada (embeddings + web) para consulta: {query}")
            
            # Inicializar modelo si es necesario
            if not self.model:
                if not self._initialize_model(api_key_path):
                    return "Lo siento, no puedo generar una respuesta en este momento debido a problemas con la configuración del modelo."
            
            # Transformar los resultados de embeddings a un formato legible
            embeddings_text = self._format_embeddings_results(search_results)
            
            # Construir prompt para combinar fuentes
            # Priorizamos la información web para consultas temporales/dinámicas
            prompt = f"""Como experto en cócteles, responde a la siguiente consulta utilizando principalmente la información web proporcionada, complementada con la información de documentos.

Consulta: {query}

Información Web (Priorizar esta información para consultas sobre eventos actuales, recetas nuevas o tendencias):
{web_info}

Información de Documentos (Información de apoyo):
{embeddings_text}

Tu respuesta debe:
1. Ser completa y precisa, priorizando la información web para eventos actuales o información temporal
2. Ser coherente y bien estructurada
3. Responder directamente a la consulta del usuario
4. Incluir detalles relevantes de ambas fuentes cuando sea posible
5. Evitar mencionar las fuentes de la información en la respuesta (no digas "según la información web" o "según la base de conocimiento")"""
            
            try:
                # Usar la biblioteca de cliente de Mistral
                from mistralai.client import MistralClient
                from mistralai.models.chat_completion import ChatMessage
                
                client = MistralClient(api_key=self.api_key)
                
                messages = [
                    ChatMessage(role="user", content=prompt)
                ]
                
                chat_response = client.chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_output_tokens
                )
                
                response_text = chat_response.choices[0].message.content
                return response_text
                    
            except ImportError:
                # Fallback a solicitud HTTP directa
                import requests
                
                logger.warning("Mistral client library not available, falling back to HTTP request")
                
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_output_tokens
                }
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return resp_json['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"Error in API response: {response.status_code} - {response.text}")
                    return f"Lo siento, ocurrió un error al consultar el modelo de lenguaje (código {response.status_code})."
                    
            except Exception as e:
                logger.error(f"Error generating combined response: {e}")
                return f"Lo siento, ocurrió un error al procesar la información: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in generate_response_from_web_and_embeddings: {e}")
            return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
    
    def _format_embeddings_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Formatea los resultados de embeddings para el prompt"""
        if not search_results:
            return "No hay resultados disponibles de la base de documentos."
            
        # Extraer contenido y metadatos relevantes
        formatted_results = []
        for i, result in enumerate(search_results[:5]):  # Limitamos a los 5 primeros resultados
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            title = metadata.get("title", f"Documento {i+1}")
            
            # Truncar contenido si es muy largo
            if len(content) > 500:
                content = content[:500] + "..."
                
            formatted_results.append(f"--- {title} ---\n{content}\n")
            
        return "\n".join(formatted_results)
