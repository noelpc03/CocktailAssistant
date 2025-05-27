"""
Generador de respuestas utilizando el modelo Gemini de Google.
"""
import os
import json
import logging
from typing import List, Dict, Any
import google.generativeai as genai

from .prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "gemini-1.5-flash" 
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_K = 32
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_OUTPUT_TOKENS = 2048

class GeminiGenerator:
    """Generate responses using Google's Gemini model."""
    
    def __init__(self, 
                 api_key_path: str = None,
                 api_key: str = None,
                 model_name: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 prompt_builder: PromptBuilder = None):
        """
        Initialize the Gemini generator.
        
        Args:
            api_key_path (str, optional): Path to file containing the API key
            api_key (str, optional): Direct API key string
            model_name (str): Model name to use
            temperature (float): Temperature for generation
            prompt_builder (PromptBuilder, optional): Custom prompt builder
        """
        # Get API key from path or direct input
        if api_key:
            self.api_key = api_key
        elif api_key_path:
            try:
                with open(api_key_path, 'r') as f:
                    # Read the token and strip any whitespace, comments or newlines
                    content = f.read()
                    # Remove commented lines or parts of lines after //
                    if '//' in content:
                        content = content.split('//')[0]
                    self.api_key = content.strip()
                    logger.info(f"API key loaded successfully from {api_key_path}")
            except Exception as e:
                logger.error(f"Failed to read API key from {api_key_path}: {str(e)}")
                raise
        else:
            raise ValueError("Either api_key or api_key_path must be provided")
            
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Store model configuration
        self.model_name = model_name
        self.temperature = temperature
        
        # Create generation configuration
        self.generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        
        try:
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            logger.info(f"Modelo principal configurado: {self.model_name}")
        except Exception as e:
            logger.warning(f"No se pudo inicializar el modelo {self.model_name}: {str(e)}")
        
        # Initialize the prompt builder
        self.prompt_builder = prompt_builder or PromptBuilder()
        
        logger.info(f"Initialized Gemini generator with model {model_name}")
        
    def generate(self, 
                query: str, 
                results: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query and search results.
        
        Args:
            query (str): User query
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            str: Generated response
        """
        try:
            logger.info(f"Generating response for query: '{query}'")
            
            # Build prompt for the model
            prompt = self.prompt_builder.build_prompt(query, results)
            
            # Prepare the full prompt with system and user parts
            full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
            
            # Try with the primary model first
            try:
                logger.info(f"Intentando generar con modelo: {self.model_name}")
                response = self.model.generate_content(full_prompt)
                generated_text = response.text
                logger.info(f"Generación exitosa con modelo: {len(generated_text)} caracteres")
                return generated_text
            except Exception as primary_error:
                # Si falla el modelo principal, intentar con el modelo de respaldo
                error_str = str(primary_error).lower()
                logger.warning(f"Error con el modelo: {error_str}")

                # Error desconocido, intentar respuesta alternativa
                return self._generate_fallback_response(query, results)
                    
        except Exception as e:
            logger.error(f"Error en proceso de generación: {str(e)}")
            return f"Lo siento, hubo un problema al generar una respuesta: {str(e)}"
                
    def _generate_fallback_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a fallback response based on the search results."""
        try:
            logger.info("Generando respuesta alternativa basada en resultados de búsqueda")
            
            # Extraer contenido relevante de los resultados
            relevant_info = ""
            for i, result in enumerate(results[:3]):  # Usar solo los 3 mejores resultados
                title = result.get("title", "Documento sin título")
                content = result.get("text", result.get("content", ""))
                
                # Tomar solo los primeros 200 caracteres de cada documento
                content_preview = content[:200] + "..." if len(content) > 200 else content
                relevant_info += f"\n\n-- Documento {i+1}: {title} --\n{content_preview}"
            
            # Crear una respuesta simple
            response = (
                f"Basado en los resultados de búsqueda encontrados, aquí hay información sobre '{query}':\n"
                f"{relevant_info}\n\n"
                f"Para más información, revise los resultados de búsqueda completos."
            )
            
            return response
        except Exception as e:
            logger.error(f"Error en respuesta alternativa: {str(e)}")
            return "No fue posible generar una respuesta. Por favor, consulte los resultados de búsqueda."
