"""
Configuration utilities for the agent system.
"""
import os
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "crawler_agent": {
        "default_urls": [
            "https://www.liquor.com/recipes/margarita/",
            "https://www.diffordsguide.com/cocktails/recipe/42/martini",
            "https://www.thespruceeats.com/classic-cocktails-everyone-should-know-760778",
            "https://en.wikipedia.org/wiki/Bartender",
        ]
    },
    "vectorizer_agent": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "retrieval_agent": {
        "max_results": 5,
        "similarity_threshold": 0.6
    },
    "search_agent": {
        "max_results": 5,
        "show_scores": True
    },
    "generation_agent": {
        "model": "mixtral-8x7b",
        "temperature": 0.4,
        "top_k": 32,
        "top_p": 0.95,
        "max_output_tokens": 2048
    },
    "coordinator_agent": {
        "timeout": 30
    }
}


class ConfigManager:
    """Configuration manager for the agent system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = DEFAULT_CONFIG.copy()
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from file if it exists"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self._merge_config(loaded_config)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Recursively merge new configuration into existing config
        
        Args:
            new_config: New configuration to merge
        """
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self._config and isinstance(self._config[key], dict):
                self._merge_config_dict(self._config[key], value)
            else:
                self._config[key] = value
    
    def _merge_config_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Merge source dict into target dict
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._merge_config_dict(target[key], value)
            else:
                target[key] = value
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
                logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type
        
        Args:
            agent_type: The type of agent to get config for
            
        Returns:
            Configuration dictionary for the agent
        """
        return self._config.get(agent_type, {})
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            new_config: New configuration to merge
        """
        self._merge_config(new_config)
        self.save_config()
