"""
Vectorization module using sentence transformers for embedding generation.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

class TextVectorizer:
    """Class for vectorizing text documents using sentence transformers."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the vectorizer with a transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Model loaded successfully")
        
    def encode_documents(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode documents into vector embeddings.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries with 'text' key
            
        Returns:
            np.ndarray: Document embeddings
        """
        texts = [doc["text"] for doc in documents]
        logger.info(f"Encoding {len(texts)} documents")
        
        embeddings = self.model.encode(texts)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode search queries into vector embeddings.
        
        Args:
            queries (List[str]): List of query strings
            
        Returns:
            np.ndarray: Query embeddings
        """
        logger.info(f"Encoding {len(queries)} queries")
        embeddings = self.model.encode(queries)
        return embeddings

def vectorize_documents(documents: List[Dict[str, Any]], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Vectorize documents using the specified model.
    
    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries with 'text' key
        model_name (str, optional): Name of the sentence transformer model
        
    Returns:
        np.ndarray: Document embeddings
    """
    vectorizer = TextVectorizer(model_name)
    return vectorizer.encode_documents(documents)
