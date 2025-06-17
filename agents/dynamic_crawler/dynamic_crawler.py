"""
Dynamic crawler that verifies responses and fetches additional information from the web when needed.
"""
import os
import requests
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class DynamicCrawler:
    """
    Dynamic crawler that verifies responses and fetches additional information from the web when needed.
    """
    
    def __init__(self, api_key_path: str = None):
        """Initialize the dynamic crawler with the API key."""
        self.api_key = None
        
        # Intentar cargar la clave API
        if not api_key_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            api_key_path = os.path.join(project_root, "tokenHuggingFace.txt")
            
        self.api_key = self._load_api_key(api_key_path)
        if not self.api_key:
            logger.error("No se pudo cargar la clave API para el crawler dinámico")
        else:
            logger.info("Clave API cargada correctamente para el crawler dinámico")
        
    def _load_api_key(self, api_key_path: str) -> str:
        """Load the API key from a file."""
        try:
            with open(api_key_path, "r") as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            return ""
    
    def verify_response(self, query: str, response: str) -> Tuple[bool, str]:
        """
        Verify if the response is correct and complete.
        Returns a tuple: (is_adequate, reason)
        """
        prompt = f"""As a cocktail expert, evaluate if this response to the query is correct and complete:

Query: {query}

Response: {response}

Analyze the response carefully and determine:
1. Is it factually CORRECT? (Are there any errors in the information provided?)
2. Is it COMPLETE? (Does it fully answer all aspects of the query?)

First, provide a brief analysis of any issues you find.
Then, provide your judgment as ONLY ONE of these exact phrases:
- "ADEQUATE" if the response is both correct and complete
- "INADEQUATE_INCORRECT" if the response contains factual errors
- "INADEQUATE_INCOMPLETE" if the response is correct but incomplete
- "INADEQUATE_BOTH" if the response has both factual errors and is incomplete

Your final judgment must be one of these EXACT phrases.
"""
        try:
            response_obj = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "mistral-medium",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            
            if response_obj.status_code == 200:
                resp_json = response_obj.json()
                full_response = resp_json['choices'][0]['message']['content'].strip()
                
                # Extract the judgment (should be at the end or in a separate line)
                lines = full_response.split('\n')
                judgment = ""
                analysis = ""
                
                # Try to find the judgment (usually at the end)
                for line in lines:
                    if any(marker in line.upper() for marker in ["ADEQUATE", "INADEQUATE"]):
                        judgment = line.strip()
                    else:
                        analysis += line + "\n"
                
                # If we couldn't find an explicit judgment, check the whole response
                if not judgment:
                    if "ADEQUATE" in full_response.upper():
                        judgment = "ADEQUATE"
                    elif "INADEQUATE_BOTH" in full_response.upper():
                        judgment = "INADEQUATE_BOTH"
                    elif "INADEQUATE_INCORRECT" in full_response.upper():
                        judgment = "INADEQUATE_INCORRECT"
                    elif "INADEQUATE_INCOMPLETE" in full_response.upper():
                        judgment = "INADEQUATE_INCOMPLETE"
                    else:
                        judgment = "INADEQUATE_INCOMPLETE"  # Default if we can't determine
                
                is_adequate = judgment.upper() == "ADEQUATE"
                logger.info(f"Respuesta verificada como {'ADECUADA' if is_adequate else 'INADECUADA'}")
                return is_adequate, analysis.strip()
            else:
                logger.error(f"Error {response_obj.status_code}: {response_obj.text}")
                return False, "Error connecting to the verification service."
                
        except Exception as e:
            logger.error(f"Exception in verify_response: {e}")
            return False, f"Exception occurred: {str(e)}"
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo and return results.
        """
        try:
            # Import here to handle potential import errors
            from duckduckgo_search import DDGS
            
            # Detectar si la consulta está en español
            is_spanish = any(word in query.lower() for word in ["cóctel", "coctel", "bebida", "siglo", "creado", "popular", "como", "qué", "cuál", "cuales", "años"])
            
            # Transform cocktail query to be more specific for better search results
            if is_spanish:
                # Consulta en español - añadir términos relevantes
                if "receta" not in query.lower():
                    if "siglo" in query.lower() or "año" in query.lower() or "década" in query.lower() or "creado" in query.lower():
                        search_query = f"cócteles {query} historia bartender"
                    else:
                        search_query = f"cóctel receta {query}"
                else:
                    search_query = query
            else:
                # Consulta en inglés
                if "recipe" not in query.lower():
                    if "century" in query.lower() or "year" in query.lower() or "decade" in query.lower() or "created" in query.lower():
                        search_query = f"cocktails {query} history bartender"
                    else:
                        search_query = f"cocktail recipe {query}"
                else:
                    search_query = query
            
            logger.info(f"Buscando en la web: {search_query}")
            
            with DDGS() as ddgs:
                # Buscar usando el idioma detectado
                results = list(ddgs.text(search_query, max_results=num_results))
            
            logger.info(f"Se encontraron {len(results)} resultados de búsqueda")
            return results
        except ImportError:
            logger.error("No se pudo importar duckduckgo_search. Asegúrate de instalar la dependencia.")
            return []
        except Exception as e:
            logger.error(f"Error searching the web: {e}")
            return []
    
    def extract_relevant_info(self, query: str, search_results: List[Dict[str, str]]) -> str:
        """
        Extract relevant information from search results using the LLM.
        """
        # Prepare search results as text
        results_text = ""
        for i, result in enumerate(search_results, 1):
            results_text += f"Source {i}:\n"
            results_text += f"Title: {result.get('title', 'No title')}\n"
            results_text += f"Snippet: {result.get('body', 'No content')}\n"
            results_text += f"URL: {result.get('href', 'No URL')}\n\n"
        
        # Determine if query is in Spanish
        is_spanish = any(word in query.lower() for word in ["cóctel", "coctel", "bebida", "siglo", "creado", "popular", "como", "qué", "cuál", "cuales", "años"])
        
        if is_spanish:
            prompt = f"""Como experto en coctelería, extrae y resume la información más relevante de estos resultados de búsqueda para responder a la consulta.

Consulta: {query}

Resultados de búsqueda:
{results_text}

Extrae solo los datos más importantes y relevantes para responder a la consulta. Concéntrate en:
- Información precisa sobre 3-4 cócteles específicos que respondan a la consulta
- Solo fechas y datos históricos esenciales si son relevantes
- Nombres concretos sin descripciones extensas

Sé conciso y directo. Resume en 3-5 puntos breves. Incluye SOLO datos precisos encontrados en los resultados proporcionados.
"""
        else:
            prompt = f"""As a cocktail expert, extract and summarize only the most essential information from these search results to answer the query.

Query: {query}

Search Results:
{results_text}

Extract only the most important and relevant data to answer the query. Focus on:
- Accurate information about 3-4 specific cocktails that answer the query
- Only essential dates and historical facts if relevant
- Concrete names without lengthy descriptions

Be concise and to-the-point. Summarize in 3-5 brief points. Include ONLY precise data found in the provided results.
"""
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "mistral-medium",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,  # Más bajo para respuestas más precisas
                    "max_tokens": 300,   # Limitado para respuestas más cortas
                    "top_p": 0.8         # Más restrictivo
                }
            )
            
            if response.status_code == 200:
                resp_json = response.json()
                info = resp_json['choices'][0]['message']['content'].strip()
                logger.info("Información relevante extraída correctamente")
                return info
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return "Error extracting information from search results."
                
        except Exception as e:
            logger.error(f"Exception in extract_relevant_info: {e}")
            return f"Error: {str(e)}"
    
    def enhance_response(self, query: str, original_response: str, web_info: str) -> str:
        """
        Create an enhanced response using the original response and web information.
        """
        # Detectar si la consulta está en español
        is_spanish = any(word in query.lower() for word in ["cóctel", "coctel", "bebida", "siglo", "creado", "popular", "como", "qué", "cuál", "cuales", "años"])
        
        # Determinar si la respuesta original está vacía o es "No se encontraron resultados"
        is_empty_response = not original_response or original_response.strip() == "" or "No se encontraron resultados" in original_response
        
        if is_spanish:
            if is_empty_response:
                # Adapt prompt based on query type
                is_historical_query = any(term in query.lower() for term in ["historia", "origen", "creó", "creado", "inventó", "inventado", "siglo", "año", "década", "cuándo", "primera vez"])
                
                # Verificar si es una consulta sobre recetas o ingredientes
                is_recipe_query = any(term in query.lower() for term in ["receta", "ingrediente", "hacer", "cómo", "preparar", "mezclar", "vaso", "copa", "servir", "adorno"])
                
                if is_historical_query:
                    prompt = f"""Como experto en coctelería e historia de cócteles, crea una respuesta concisa a esta consulta histórica usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Crea una respuesta breve y precisa que:
1. Responda directamente a la consulta con los datos más relevantes
2. Mencione exactamente qué cócteles cumplen con los criterios de la consulta (máximo 3-4 ejemplos)
3. Sea concisa y directa, evitando explicaciones largas
4. Si la consulta es sobre historia o características de cócteles, proporciona fechas y orígenes precisos

Para consultas temporales (historia, origen, etc.), enfócate en datos históricos concretos.
Para consultas sobre recetas o ingredientes, menciona componentes exactos y proporciones.

La respuesta debe tener un máximo de 5-6 oraciones, estar en español y ser fácil de entender."""
                elif is_recipe_query:
                    prompt = f"""Como bartender profesional, crea una respuesta concisa a esta consulta sobre recetas o ingredientes usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Crea una respuesta breve y precisa que:
1. Responda directamente a la consulta con información precisa sobre la receta
2. Liste los ingredientes exactos y sus proporciones para los cócteles relevantes
3. Mencione técnicas específicas de preparación y el vaso o copa adecuado
4. Se enfoque en 1-2 cócteles en detalle en lugar de varios superficialmente

Para consultas sobre ingredientes, especifica los ingredientes exactos y posibles sustitutos.
Para consultas sobre recetas, proporciona instrucciones claras paso a paso.
Para preguntas sobre cristalería o servicio, sé específico sobre los recipientes apropiados y la presentación.

La respuesta debe tener un máximo de 5-6 oraciones, estar en español y ser fácil de entender."""
                else:
                    prompt = f"""Como experto en coctelería, crea una respuesta concisa a esta consulta usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Crea una respuesta breve y precisa que:
1. Responda directamente a la consulta con los datos más relevantes
2. Mencione exactamente qué cócteles cumplen con los criterios de la consulta (máximo 3-4 ejemplos)
3. Sea concisa y directa, evitando explicaciones largas
4. Si la consulta es sobre características de cócteles, proporcione detalles específicos

Enfócate en proporcionar información precisa y factual en lugar de descripciones generales.
Incluye ejemplos específicos que coincidan exactamente con los criterios de la consulta.

La respuesta debe tener un máximo de 5-6 oraciones, estar en español y ser fácil de entender."""
            else:
                prompt = f"""Como experto en coctelería, proporciona una versión mejorada y concisa de esta respuesta.

Consulta: {query}

Respuesta Original: {original_response}

Información Adicional: {web_info}

Crea una respuesta breve que:
1. Corrija cualquier error factual de la respuesta original
2. Incluya solo los datos más relevantes para la consulta
3. Sea concisa y directa, limitada a 5-6 oraciones

Enfócate en responder exactamente lo que se pregunta sin información adicional innecesaria."""
        else:
            if is_empty_response:
                # Adapt prompt based on query type
                is_historical_query = any(term in query.lower() for term in ["history", "origin", "created", "invented", "century", "year", "decade", "when", "first time"])
                is_recipe_query = any(term in query.lower() for term in ["recipe", "ingredient", "make", "how to", "prepare", "mix", "glass", "garnish"])
                
                if is_historical_query:
                    prompt = f"""As a cocktail expert and historian, create a concise answer to this historical query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please create a brief response that:
1. Directly answers the query with precise historical facts
2. Includes exact dates, locations, and historical context
3. Mentions exactly which cocktails meet the criteria of the query (max 3-4 examples)
4. Is concise and to-the-point, avoiding lengthy explanations

For historical queries, focus on concrete dates, origins, and evolution.
For recipe or ingredient queries, include specific components and proportions.

The response should be limited to 5-6 sentences maximum."""
                elif is_recipe_query:
                    prompt = f"""As a professional bartender, create a concise answer to this recipe or ingredient query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please create a brief response that:
1. Directly answers the query with precise recipe information
2. Lists the exact ingredients and proportions for relevant cocktails
3. Mentions specific preparation techniques and serving vessels
4. Focuses on 1-2 cocktails in detail rather than many superficially

For ingredient queries, specify exact ingredients and possible substitutions.
For recipe queries, provide clear step-by-step instructions.
For glassware or service questions, be specific about the proper vessels and presentation.

The response should be limited to 5-6 sentences maximum."""
                else:
                    prompt = f"""As a cocktail expert, create a concise answer to this query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please create a brief response that:
1. Directly answers the query with the most relevant facts
2. Mentions exactly which cocktails meet the criteria of the query (max 3-4 examples)
3. Is concise and to-the-point, avoiding lengthy explanations
4. If the query is about cocktail characteristics, provide specific details

Focus on providing precise, factual information rather than general descriptions.
Include specific examples that exactly match the query criteria.

The response should be limited to 5-6 sentences maximum."""
            else:
                prompt = f"""As a cocktail expert, provide an improved and concise version of this response.

Query: {query}

Original Response: {original_response}

Additional Information: {web_info}

Create a brief response that:
1. Corrects any factual errors in the original response
2. Includes only the most relevant information for the query
3. Is concise and direct, limited to 5-6 sentences

Focus on answering exactly what was asked without unnecessary additional information."""

        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "mistral-medium",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 350,  # Reducido para respuestas más cortas
                    "top_p": 0.8  # Más restrictivo para respuestas más enfocadas
                }
            )
            
            if response.status_code == 200:
                resp_json = response.json()
                enhanced = resp_json['choices'][0]['message']['content'].strip()
                logger.info("Respuesta mejorada generada correctamente")
                return enhanced
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return original_response + "\n\nNote: Additional information could not be retrieved."
                
        except Exception as e:
            logger.error(f"Exception in enhance_response: {e}")
            return original_response + f"\n\nNote: Error enhancing response: {str(e)}"
    
    def generate_response_from_scratch(self, query: str, web_info: str) -> str:
        """
        Generate a response from scratch using the web information when there is no
        original response to enhance.
        
        Args:
            query: User query
            web_info: Information extracted from web search results
            
        Returns:
            Generated response or error message
        """
        try:
            # Determine if query is in Spanish
            is_spanish = any(word in query.lower() for word in ["cóctel", "coctel", "bebida", "siglo", "creado", "popular", "como", "qué", "cuál", "cuales", "años"])
            
            # Check if this is a recipe or history query
            is_recipe_query = any(word in query.lower() for word in ["receta", "recipe", "preparar", "hacer", "ingredientes", "ingredients", "mezclar", "mix"])
            is_history_query = any(word in query.lower() for word in ["historia", "history", "origen", "origin", "invented", "creado", "siglo", "century", "año", "year", "década", "decade"])
            
            # Generate the prompt based on query type and language
            if is_spanish:
                if is_history_query:
                    prompt = f"""Como historiador de coctelería, crea una respuesta completa a esta consulta histórica usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Proporciona una respuesta detallada que:
1. Responda directamente a la consulta con datos históricos precisos
2. Incluya fechas exactas, lugares de origen, y personas involucradas en la creación
3. Mencione la evolución histórica del cóctel si es relevante
4. Proporcione contexto cultural o social cuando sea apropiado

Enfócate en hechos históricos concretos y verificables, evitando especulaciones.
La respuesta debe estar en español y ser informativa pero concisa."""
                elif is_recipe_query:
                    prompt = f"""Como maestro mixólogo, crea una respuesta completa a esta consulta sobre recetas usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Proporciona una respuesta detallada que:
1. Liste todos los ingredientes necesarios con sus proporciones exactas
2. Explique el paso a paso de la preparación de manera clara
3. Especifique el tipo de vaso o copa adecuado y la decoración recomendada
4. Mencione cualquier variación popular si es relevante

Asegúrate de incluir consejos prácticos que ayuden a preparar mejor el cóctel.
La respuesta debe estar en español y ser informativa pero concisa."""
                else:
                    prompt = f"""Como experto en coctelería, crea una respuesta completa a esta consulta usando la información obtenida de la web.

Consulta: {query}

Información de la Web: {web_info}

Proporciona una respuesta detallada que:
1. Responda directamente a la consulta con información precisa y relevante
2. Incluya datos específicos sobre los cócteles mencionados
3. Proporcione información sobre sabores, ingredientes clave y características distintivas
4. Responda todos los aspectos de la consulta de manera completa

La respuesta debe estar en español, ser informativa pero concisa, y estar basada únicamente en los datos proporcionados."""
            else:
                if is_history_query:
                    prompt = f"""As a cocktail historian, create a comprehensive answer to this historical query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please provide a detailed response that:
1. Directly answers the query with precise historical data
2. Includes exact dates, places of origin, and people involved in the creation
3. Mentions the historical evolution of the cocktail if relevant
4. Provides cultural or social context when appropriate

Focus on concrete and verifiable historical facts, avoiding speculation.
The response should be informative yet concise."""
                elif is_recipe_query:
                    prompt = f"""As a master mixologist, create a comprehensive answer to this recipe query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please provide a detailed response that:
1. Lists all necessary ingredients with their exact proportions
2. Explains the step-by-step preparation clearly
3. Specifies the appropriate type of glass and recommended garnish
4. Mentions any popular variations if relevant

Make sure to include practical tips that help prepare the cocktail better.
The response should be informative yet concise."""
                else:
                    prompt = f"""As a cocktail expert, create a comprehensive answer to this query using information gathered from the web.

Query: {query}

Information from Web: {web_info}

Please provide a detailed response that:
1. Directly answers the query with accurate and relevant information
2. Includes specific data about the mentioned cocktails
3. Provides information about flavors, key ingredients, and distinctive characteristics
4. Addresses all aspects of the query comprehensively

The response should be informative yet concise and based solely on the provided data."""
            
            # Call LLM API to generate the response
            return self._call_llm(prompt)
            
        except Exception as e:
            logger.error(f"Exception in generate_response_from_scratch: {e}")
            return f"Lo siento, no pude generar una respuesta debido a: {str(e)}. Por favor, intenta una consulta más específica."
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the language model API with the given prompt.
        """
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "mistral-medium",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                resp_json = response.json()
                return resp_json['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return "Error calling the language model API."
                
        except Exception as e:
            logger.error(f"Exception in _call_llm: {e}")
            return f"Error: {str(e)}"
