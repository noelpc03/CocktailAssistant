"""
Shared data storage for agents to access common resources.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional
import threading

logger = logging.getLogger(__name__)

class DataStore:
    """
    Singleton data store for agents to share information.
    Thread-safe implementation for shared state.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataStore, cls).__new__(cls)
                cls._instance._data = {}
                # Define base project path
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # Define primary and fallback paths
                primary_embeddings_dir = os.path.join(project_root, "agents", "data", "embeddings")
                fallback_embeddings_dir = os.path.join(project_root, "src", "data", "embeddings")
                
                # Use primary path if files exist, otherwise use fallback
                if (os.path.exists(os.path.join(primary_embeddings_dir, "faiss_index.bin")) and
                    os.path.exists(os.path.join(primary_embeddings_dir, "documents_metadata.json"))):
                    embeddings_dir = primary_embeddings_dir
                    logger.info(f"Using primary embeddings directory: {primary_embeddings_dir}")
                elif (os.path.exists(os.path.join(fallback_embeddings_dir, "faiss_index.bin")) and
                    os.path.exists(os.path.join(fallback_embeddings_dir, "documents_metadata.json"))):
                    embeddings_dir = fallback_embeddings_dir
                    logger.info(f"Using fallback embeddings directory: {fallback_embeddings_dir}")
                else:
                    embeddings_dir = primary_embeddings_dir
                    logger.warning(f"No valid embeddings found. Using default directory: {embeddings_dir}")
                
                cls._instance._file_paths = {
                    "embeddings_dir": embeddings_dir,
                    "metadata_path": os.path.join(project_root, "agents", "data", "metadata.json"),
                }
                # Ensure directories exist
                cls._instance._ensure_directories()
        return cls._instance
    
    def _ensure_directories(self) -> None:
        """Ensure that necessary directories exist"""
        os.makedirs(self._file_paths["embeddings_dir"], exist_ok=True)
        
    def get(self, key: str, default=None) -> Any:
        """
        Get a value from the data store
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The stored value or default
        """
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the data store
        
        Args:
            key: The key to store under
            value: The value to store
        """
        with self._lock:
            self._data[key] = value
            logger.debug(f"DataStore: Set {key}")
    
    def delete(self, key: str) -> None:
        """
        Delete a key from the data store
        
        Args:
            key: The key to delete
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                logger.debug(f"DataStore: Deleted {key}")
    
    def list_keys(self) -> List[str]:
        """
        List all keys in the data store
        
        Returns:
            List of keys in the store
        """
        with self._lock:
            return list(self._data.keys())
    
    def get_file_path(self, key: str) -> str:
        """
        Get a predefined file path
        
        Args:
            key: The file path key
            
        Returns:
            The file path string
        """
        return self._file_paths.get(key, "")
    
    def save_metadata(self, documents: List[Dict[str, Any]]) -> None:
        """
        Save document metadata to a JSON file
        
        Args:
            documents: List of document metadata to save
        """
        path = self._file_paths["metadata_path"]
        with open(path, "w") as f:
            json.dump(documents, f, indent=2)
        logger.info(f"Saved metadata for {len(documents)} documents to {path}")
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load document metadata from JSON file
        
        Returns:
            List of document metadata
        """
        path = self._file_paths["metadata_path"]
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []
