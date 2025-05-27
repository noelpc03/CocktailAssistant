"""
Query processor module for processing natural language search queries.
"""
import re
from typing import List, Dict, Any
import numpy as np
import logging

# Reuse the existing vectorizer
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from retrieval.embeddings.vectorizer import TextVectorizer, DEFAULT_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    """Process and vectorize natural language queries."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the query processor.
        
        Args:
            model_name (str): The sentence transformer model to use
        """
        logger.info(f"Initializing query processor with model: {model_name}")
        self.vectorizer = TextVectorizer(model_name)
        
    def process_query(self, query: str) -> str:
        """
        Clean and normalize the query text.
        
        Args:
            query (str): Raw query text
            
        Returns:
            str: Processed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        logger.info(f"Processed query: '{query}'")
        return query
        
    def vectorize_query(self, query: str) -> np.ndarray:
        """
        Convert query to vector embedding.
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding vector
        """
        # Process the query
        processed_query = self.process_query(query)
        
        # Vectorize using the existing TextVectorizer
        query_embedding = self.vectorizer.encode_queries([processed_query])[0]
        
        logger.info(f"Vectorized query with shape {query_embedding.shape}")
        return query_embedding