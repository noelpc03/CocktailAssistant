"""
Retrieval engine for searching documents using query embeddings.
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os

# Import the existing FAISS storage
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from retrieval.storage.faiss_store import FAISSStorage

# Define embeddings directory
EMBEDDINGS_DIR = os.path.join(parent_dir, "data", "embeddings")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEngine:
    """Engine to search for documents using vectorized queries."""
    
    def __init__(self, index_directory: str = EMBEDDINGS_DIR):
        """
        Initialize the retrieval engine with a stored FAISS index.
        
        Args:
            index_directory (str): Directory containing the FAISS index
        """
        logger.info(f"Initializing retrieval engine from {index_directory}")
        try:
            self.storage = FAISSStorage.load(index_directory)
            logger.info(f"Loaded index with {len(self.storage.documents)} documents")
        except FileNotFoundError as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query embedding.
        
        Args:
            query_embedding (np.ndarray): Query vector
            top_k (int): Maximum number of results to return
            score_threshold (float, optional): Maximum score threshold
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with scores
        """
        # Use the existing search method from FAISSStorage
        results = self.storage.search(query_embedding, top_k)
        
        # Apply threshold filtering if specified
        if score_threshold is not None:
            results = [doc for doc in results if doc.get("score", float('inf')) <= score_threshold]
            
        logger.info(f"Found {len(results)} matching documents for query")
        return results