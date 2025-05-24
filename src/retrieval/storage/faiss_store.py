"""
FAISS vector storage module for document embeddings.
"""
import faiss
import numpy as np
import os
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSStorage:
    """Class for storing and retrieving vector embeddings using FAISS."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS storage.
        
        Args:
            dimension (int): Dimension of embeddings 
        """
        logger.info(f"Initializing FAISS index with dimension {dimension}")
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        
    def add_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            documents (List[Dict[str, Any]]): Original document data
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
            
        logger.info(f"Adding {len(embeddings)} embeddings to FAISS index")
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store original documents
        self.documents.extend(documents)
        
        logger.info(f"FAISS index now contains {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no results to return")
            return []
            
        # Ensure the query is in the right shape for FAISS
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get matching documents
        results = []
        for (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(dist)
                results.append(doc)
        
        return results
        
    def save(self, directory: str) -> None:
        """
        Save the FAISS index and documents to disk.
        
        Args:
            directory (str): Directory to save the index and metadata
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
        metadata_path = os.path.join(directory, "documents_metadata.json")
        logger.info(f"Saving documents metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(self.documents, f)
            
        logger.info(f"Storage saved successfully")
        
    @classmethod
    def load(cls, directory: str) -> 'FAISSStorage':
        """
        Load a FAISS index and documents from disk.
        
        Args:
            directory (str): Directory containing the saved index and metadata
            
        Returns:
            FAISSStorage: Loaded storage object
        """
        index_path = os.path.join(directory, "faiss_index.bin")
        metadata_path = os.path.join(directory, "documents_metadata.json")
        
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError(f"Index or metadata not found in {directory}")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        # Load documents metadata
        logger.info(f"Loading documents metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            documents = json.load(f)
            
        # Create and populate storage object
        storage = cls(dimension=index.d)
        storage.index = index
        storage.documents = documents
        
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors and {len(documents)} documents")
        return storage

def store_embeddings(embeddings: np.ndarray, documents: List[Dict[str, Any]], 
                    save_directory: Optional[str] = None) -> FAISSStorage:
    """
    Store embeddings in FAISS and optionally save to disk.
    
    Args:
        embeddings (np.ndarray): Document embeddings
        documents (List[Dict[str, Any]]): Original document data
        save_directory (str, optional): Directory to save the index and metadata
        
    Returns:
        FAISSStorage: Storage object with embeddings
    """
    # Get dimensionality from embeddings
    dimension = embeddings.shape[1]
    
    # Create storage and add embeddings
    storage = FAISSStorage(dimension=dimension)
    storage.add_embeddings(embeddings, documents)
    
    # Save if directory provided
    if save_directory:
        storage.save(save_directory)
        
    return storage
