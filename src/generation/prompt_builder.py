"""
Construcción de prompts para el modelo de lenguaje.
"""
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptBuilder:
    """Class to build prompts for the language model based on retrieved documents."""
    
    def __init__(self, 
                 system_prompt: str = None, 
                 max_context_length: int = 100000):
        """
        Initialize the prompt builder.
        
        Args:
            system_prompt (str, optional): Template for the system prompt
            max_context_length (int): Maximum context length allowed
        """
        self.max_context_length = max_context_length
        
        # Default system prompt if none provided
        if system_prompt is None:
            self.system_prompt = (
                "Eres un asistente experto en bartenders, cócteles y bebidas. "
                "Tu objetivo es responder preguntas basándote solamente en la información proporcionada. "
                "Si la información no es suficiente para responder con precisión, indícalo. "
                "Sé claro, preciso y útil en tus respuestas. "
                "IMPORTANTE: NUNCA menciones los documentos o fuentes específicas en tu respuesta. "
                "No debes indicar de qué documento proviene la información, solo responde a la pregunta directamente. "
                "La respuesta debe ser en el idioma de la pregunta del usuario y los documentos proporcionados están en español o en inglés. "
            )
        else:
            self.system_prompt = system_prompt
            
        logger.info("Initialized prompt builder")
    
    def _extract_relevant_content(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Extract relevant content from search results.
        
        Args:
            results (List[Dict[str, Any]]): Retrieved documents
            query (str): The original query
            
        Returns:
            str: Concatenated relevant content
        """
        contexts = []
        current_length = 0
        
        # Add context from each result, respecting the max length
        for i, result in enumerate(results):
            # Extract content, title and URL
            content = result.get("text", result.get("content", ""))
            title = result.get("title", "Documento sin título")
            url = result.get("url", "URL no disponible")
            
            # Format the context
            
            doc_context = f"{content}\n\n"
            
            # Check if adding this document exceeds the max length
            if current_length + len(doc_context) > self.max_context_length:
                # If it's the first document, truncate it
                if i == 0:
                    truncated = doc_context[:self.max_context_length]
                    contexts.append(truncated)
                break
            
            # Add the document context
            contexts.append(doc_context)
            current_length += len(doc_context)
        
        return "\n".join(contexts)
    
    def build_prompt(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Build a prompt for the language model.
        
        Args:
            query (str): The user query
            results (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            Dict[str, str]: Formatted prompt ready for the model
        """
        logger.info(f"Building prompt for query: '{query}'")
        
        # Extract relevant information from results
        context = self._extract_relevant_content(results, query)
        
        # Format the user message
        user_message = (
            f"Basándote en la siguiente información sobre bartenders y cócteles, "
            f"responde a esta pregunta: {query}\n\n"
            f"Información disponible:\n{context}"
        )
        
        # Format the complete prompt
        prompt = {
            "system": self.system_prompt,
            "user": user_message
        }
        
        logger.info(f"Built prompt with {len(context)} characters of context")
        return prompt
