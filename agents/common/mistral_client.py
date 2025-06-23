"""
MistralClient: Simple client for interacting with the Mistral AI API
"""

import os
import logging
import json
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MistralClient:
    """A simple client for interacting with the Mistral AI API"""
    
    def __init__(self, api_key: str = None, model_name: str = "mistral-medium"):
        """
        Initialize the Mistral AI client.
        
        Args:
            api_key: The API key to use (if None, will try to load from environment or file)
            model_name: The model to use (default: mistral-medium)
        """
        self.api_key = api_key or self._load_api_key()
        self.model_name = model_name
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        
        # Default generation parameters
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 1024
        
        if not self.api_key:
            logger.warning("No API key provided for Mistral AI. LLM-based features will be disabled.")
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load the API key from environment variable or token file.
        
        Returns:
            The API key if found, None otherwise
        """
        # First try environment variable
        api_key = os.environ.get("MISTRAL_API_KEY")
        if api_key:
            return api_key
            
        # Try to load from token file in project root
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            token_file = os.path.join(project_root, "tokenHuggingFace.txt")
            
            if os.path.exists(token_file):
                with open(token_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    return api_key
        except Exception as e:
            logger.error(f"Error loading API key from file: {e}")
        
        return None
    
    def set_parameters(self, temperature: float = None, top_p: float = None, 
                      max_tokens: int = None) -> None:
        """
        Set generation parameters.
        
        Args:
            temperature: Sampling temperature (default: 0.7)
            top_p: Top-p sampling parameter (default: 0.9)
            max_tokens: Maximum number of tokens to generate (default: 1024)
        """
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    async def generate(self, prompt: str, system_message: str = None) -> Dict[str, Any]:
        """
        Generate a response from the Mistral AI model.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message for ChatML format
            
        Returns:
            Response dictionary with 'text' field containing the generated text
        """
        if not self.api_key:
            return {"text": "", "error": "No API key available"}
        
        try:
            # Prepare messages in ChatML format
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
                
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "safe_prompt": True
            }
            
            # Make the request with a longer timeout (45 seconds)
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Log success for debugging
            logger.info(f"Received successful response from Mistral API with {len(response_data.get('choices', []))} choices")
            
            # Handle missing or empty choices
            if not response_data.get("choices"):
                logger.warning("Empty choices in Mistral API response")
                return {"text": "", "error": "Empty response from Mistral API"}
                
            # Extract the generated text
            generated_text = response_data["choices"][0]["message"]["content"]
            
            # Return the result
            return {
                "text": generated_text,
                "finish_reason": response_data["choices"][0].get("finish_reason", "unknown"),
                "model": response_data.get("model", self.model_name),
                "usage": response_data.get("usage", {})
            }
            
        except Exception as e:
            logger.error(f"Error generating response with Mistral AI: {e}")
            return {"text": "", "error": str(e)}
