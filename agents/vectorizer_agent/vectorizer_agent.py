"""
Vectorizer agent responsible for generating embeddings from documents.
"""
import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class VectorizerAgent(Agent):
    """Agent responsible for creating vector embeddings of documents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the vectorizer agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("vectorizer_agent")
        self.data_store = DataStore()
        self.model_name = self.config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
        self.chunk_size = self.config.get("chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        
        # Load model and tokenizer lazily when needed
        self.model = None
        self.tokenizer = None
        
    async def start(self) -> None:
        """Start the vectorizer agent"""
        await super().start()
        logger.info(f"Vectorizer agent {self.agent_id} started")
    
    def _load_model(self) -> None:
        """Load the transformer model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading model: {self.model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                # Put model in evaluation mode
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("Using GPU for embeddings")
                else:
                    logger.info("Using CPU for embeddings")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        content = message.get("content", {})
        action = content.get("action")
        
        if action == "vectorize_documents":
            documents = content.get("documents", [])
            operation_id = content.get("operation_id")
            
            logger.info(f"Received vectorize_documents with operation_id: {operation_id}")
            
            if not documents:
                documents = self.data_store.get("crawled_documents", [])
                
            if not documents:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "vectorize_results",
                        "operation_id": operation_id,
                        "status": "error",
                        "message": "No documents to vectorize"
                    }
                }
            
            chunked_docs, embeddings = await self.vectorize_documents(documents)
            
            # Store the results
            self.data_store.set("chunked_documents", chunked_docs)
            self.data_store.set("document_embeddings", embeddings)
            
            logger.info(f"Vectorization completed for operation_id: {operation_id}")
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "vectorize_results",
                    "operation_id": operation_id,
                    "status": "success",
                    "num_chunks": len(chunked_docs),
                    "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
                }
            }
            
        return None
    
    async def vectorize_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Convert documents to vector embeddings.
        
        Args:
            documents: List of documents to vectorize
            
        Returns:
            Tuple of (chunked documents, embeddings matrix)
        """
        # Load model if not already loaded
        self._load_model()
        
        # Chunk documents
        chunked_docs = self._chunk_documents(documents)
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Generate embeddings
        embeddings = []
        for chunk in tqdm(chunked_docs, desc="Generating embeddings"):
            embedding = self._generate_embedding(chunk["text"])
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings)
        
        logger.info(f"Generated embeddings with shape {embeddings_matrix.shape}")
        return chunked_docs, embeddings_matrix
    
    def _chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for better embeddings.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunks = []
        
        for doc in documents:
            doc_text = doc.get("content", "")
            doc_title = doc.get("title", "")
            doc_url = doc.get("url", "")
            
            # Skip empty documents
            if not doc_text.strip():
                continue
            
            # Split into paragraphs first
            paragraphs = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
            
            current_chunk = ""
            current_position = 0
            
            for paragraph in paragraphs:
                # If adding paragraph would exceed chunk size and we already have content
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    # Save the current chunk
                    chunk_doc = {
                        "doc_id": len(chunks),
                        "text": current_chunk,
                        "title": doc_title,
                        "source_url": doc_url,
                        "position": current_position
                    }
                    chunks.append(chunk_doc)
                    
                    # Start a new chunk with overlap
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                    current_chunk = " ".join(overlap_words) + " " + paragraph
                    current_position += 1
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += " " + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if not empty
            if current_chunk:
                chunk_doc = {
                    "doc_id": len(chunks),
                    "text": current_chunk,
                    "title": doc_title,
                    "source_url": doc_url,
                    "position": current_position
                }
                chunks.append(chunk_doc)
        
        return chunks
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text chunk.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        try:
            # Truncate text if needed
            if len(text) > 10000:
                text = text[:10000]
                
            # Encode text
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                
                # Move inputs to GPU if available
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                outputs = self.model(**inputs)
                
                # Use mean of last hidden state as embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                # Normalize embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768)  # Default embedding size
