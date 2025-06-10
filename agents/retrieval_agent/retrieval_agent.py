"""
Retrieval agent responsible for storing and querying vector embeddings.
"""
import os
import json
import numpy as np
import faiss
import logging
from typing import Dict, Any, List, Tuple
import time

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class RetrievalAgent(Agent):
    """Agent responsible for vector storage and retrieval"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the retrieval agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("retrieval_agent")
        self.data_store = DataStore()
        self.max_results = self.config.get("max_results", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.2)  # Lowered threshold
        self.index = None
        self.embeddings_dir = self.data_store.get_file_path("embeddings_dir")
        
        # File paths
        self.index_file = os.path.join(self.embeddings_dir, "faiss_index.bin")
        self.metadata_file = os.path.join(self.embeddings_dir, "documents_metadata.json")
    
    async def start(self) -> None:
        """Start the retrieval agent"""
        await super().start()
        
        # Try to load existing index if exists
        self._load_index_if_exists()
        
        logger.info(f"Retrieval agent {self.agent_id} started")
    
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
        
        if action == "store_embeddings":
            chunked_docs = content.get("documents") or self.data_store.get("chunked_documents")
            embeddings = content.get("embeddings") or self.data_store.get("document_embeddings")
            operation_id = content.get("operation_id")
            
            logger.info(f"Received store_embeddings with operation_id: {operation_id}")
            
            if chunked_docs is None or embeddings is None:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "store_results",
                        "operation_id": operation_id,
                        "status": "error",
                        "message": "Missing documents or embeddings data"
                    }
                }
                
            success = await self.store_embeddings(chunked_docs, embeddings)
            
            logger.info(f"Storage completed for operation_id: {operation_id} with success={success}")
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "store_results",
                    "operation_id": operation_id,
                    "status": "success" if success else "error"
                }
            }
            
        elif action == "query_embeddings":
            query_vector = content.get("query_vector")
            top_k = content.get("top_k", self.max_results)
            
            # Verificar si tenemos índice antes de proceder
            if not os.path.exists(self.index_file):
                logger.error(f"Index file not found at {self.index_file}")
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "status": "error",
                        "message": "No index available - please run crawl first",
                        "results": []
                    }
                }
            
            original_sender = content.get("original_sender")
            query = content.get("query", "")
            
            # Obtener la consulta del data_store para búsqueda de texto
            if query:
                self.data_store.set("current_query", query)
            
            if query_vector is None:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "status": "error",
                        "message": "Missing query vector",
                        "original_sender": original_sender,
                        "query": query
                    }
                }
                
            results = await self.query_index(query_vector, top_k)
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "query_results",
                    "status": "success",
                    "results": results,
                    "original_sender": original_sender,
                    "query": query
                }
            }
            
        return None
    
    async def store_embeddings(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """
        Store document embeddings in FAISS index.
        
        Args:
            documents: Document chunks metadata
            embeddings: Numpy array of embeddings
            
        Returns:
            Success flag
        """
        try:
            if len(documents) != embeddings.shape[0]:
                logger.error(f"Mismatch between documents ({len(documents)}) and embeddings ({embeddings.shape[0]})")
                return False
                
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            
            # Asignar doc_id a cada documento y registrar en logs
            for i, doc in enumerate(documents):
                if "doc_id" not in doc:
                    doc["doc_id"] = i
                    logger.info(f"Added doc_id {i} to document with title: {doc.get('title', 'No title')}")
                    
            # Create a new FAISS index using L2 distance like the working version
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance (same as working version)
            
            # Add vectors to the index
            self.index = faiss.IndexIDMap(self.index)
            
            # Normalizar los vectores
            normalized_embeddings = embeddings.copy().astype('float32')
            for i in range(len(normalized_embeddings)):
                norm = np.linalg.norm(normalized_embeddings[i])
                if norm > 0:
                    normalized_embeddings[i] = normalized_embeddings[i] / norm
            
            # Asegurarse que los IDs sean enteros
            doc_ids = np.array([int(doc["doc_id"]) for doc in documents], dtype=np.int64)
            self.index.add_with_ids(normalized_embeddings, doc_ids)
            
            # Save the index
            os.makedirs(self.embeddings_dir, exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            logger.info(f"FAISS index saved to {self.index_file}")
            
            # Save document metadata
            with open(self.metadata_file, "w") as f:
                json.dump(documents, f, indent=2)
            logger.info(f"Document metadata saved to {self.metadata_file}")
            
            # Save to data store for other agents to use
            self.data_store.save_metadata(documents)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def _load_index_if_exists(self) -> bool:
        """
        Load FAISS index from disk if it exists.
        
        Returns:
            Success flag
        """
        try:
            if os.path.exists(self.index_file):
                logger.info(f"Loading FAISS index from {self.index_file}")
                self.index = faiss.read_index(self.index_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def _load_document_metadata(self) -> List[Dict[str, Any]]:
        """
        Load document metadata from disk.
        
        Returns:
            List of document metadata
        """
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    documents = json.load(f)
                    
                    # Add doc_id if not present (using index as ID)
                    for i, doc in enumerate(documents):
                        if "doc_id" not in doc:
                            doc["doc_id"] = i
                            
                    return documents
            return []
        except Exception as e:
            logger.error(f"Error loading document metadata: {e}")
            return []
    
    async def query_index(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the FAISS index with a vector.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        try:
            if self.index is None:
                if not self._load_index_if_exists():
                    logger.error("No index available for querying")
                    return []
                    
            # Make sure query vector is properly shaped and normalized
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # Verificar y ajustar la dimensión del vector de consulta si es necesario
            index_dimension = self.index.d
            query_dimension = query_vector.shape[1]
            
            if index_dimension != query_dimension:
                logger.warning(f"Dimension mismatch: index={index_dimension}, query={query_dimension}")
                
                # Este es un problema crítico - vamos a hacer una búsqueda por texto en su lugar
                documents = self._load_document_metadata()
                query_text = self.data_store.get("current_query", "")
                
                if query_text and documents:
                    # Buscar por texto en lugar de por vector
                    logger.info(f"Falling back to text search for query: {query_text}")
                    results = self._text_based_search(query_text, documents)
                    if results:
                        return results
                        
                # Si no hay resultados por texto, devolvemos lista vacía
                return []
                
            # Normalizar el vector de consulta
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
                
            # Cast to the correct data type (float32) for FAISS
            query_vector = query_vector.astype('float32')
            
            # Perform the search
            start_time = time.time()
            distances, doc_ids = self.index.search(query_vector, top_k)
            query_time = time.time() - start_time
            
            # Flatten results
            distances = distances[0]
            doc_ids = doc_ids[0]
            
            # Cargar metadatos de documentos y asegurar que cada documento tenga un doc_id
            documents = self._load_document_metadata()
            
            # Crear mapeo tanto por índice como por doc_id (si existe)
            doc_map = {}
            for i, doc in enumerate(documents):
                if "doc_id" in doc:
                    doc_map[doc["doc_id"]] = doc
                doc_map[i] = doc  # Usar índice como respaldo
            
            logger.info(f"Doc map has {len(doc_map)} entries")
            
            # Umbral muy bajo para permitir más resultados
            self.similarity_threshold = 0.01
            
            results = []
            for i, doc_id in enumerate(doc_ids):
                if doc_id != -1:
                    score = float(distances[i])
                    # Para distancia L2, menor es mejor, convertimos a similitud
                    similarity = 1.0 / (1.0 + score)
                    
                    logger.info(f"Doc {doc_id}: raw distance={score:.4f}, similarity={similarity:.4f}")
                    
                    # Intentar buscar el documento por doc_id o por índice
                    if doc_id in doc_map:
                        doc = doc_map[doc_id].copy()
                        doc["score"] = similarity
                        results.append(doc)
                    elif i < len(documents):
                        # Usar el índice como respaldo
                        doc = documents[i].copy()
                        doc["score"] = similarity
                        doc["doc_id"] = i  # Asignar doc_id si no existe
                        results.append(doc)
                        logger.info(f"Using fallback index {i} for doc_id {doc_id}")
            
            # Si no hay resultados, intenta búsqueda por texto
            if not results:
                query_text = self.data_store.get("current_query", "")
                if query_text:
                    text_results = self._text_based_search(query_text, documents)
                    if text_results:
                        results = text_results
            
            logger.info(f"Query returned {len(results)} results in {query_time:.4f}s (threshold={self.similarity_threshold})")
            return results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error querying index: {e}\n{error_details}")
            
            # Intenta hacer búsqueda por texto como último recurso
            try:
                documents = self._load_document_metadata()
                query_text = self.data_store.get("current_query", "")
                if query_text and documents:
                    results = self._text_based_search(query_text, documents)
                    if results:
                        logger.info(f"Fallback text search found {len(results)} results")
                        return results
            except Exception as text_search_error:
                logger.error(f"Text search fallback also failed: {text_search_error}")
                
            return []
            
    def _text_based_search(self, query_text: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Realiza una búsqueda basada en texto.
        
        Args:
            query_text: El texto de búsqueda
            documents: Lista de documentos para buscar
            
        Returns:
            Lista de documentos que coinciden con la búsqueda
        """
        logger.info(f"Performing text-based search for: {query_text}")
        
        # Convertir la consulta a minúsculas y dividir en términos
        query_terms = query_text.lower().split()
        results = []
        
        for doc in documents:
            title = doc.get("title", "").lower()
            text = doc.get("text", "").lower()
            
            # Buscar coincidencias exactas primero (más alta prioridad)
            if query_text.lower() in title or query_text.lower() in text:
                doc_copy = doc.copy()
                doc_copy["score"] = 0.9  # Alta puntuación para coincidencias exactas
                results.append(doc_copy)
                logger.info(f"Found exact match in document: {doc.get('title')}")
                continue
                
            # Buscar coincidencias parciales (términos individuales)
            match_count = sum(1 for term in query_terms if term in title or term in text)
            if match_count > 0:
                match_ratio = match_count / len(query_terms)
                if match_ratio > 0.5:  # Al menos la mitad de los términos coinciden
                    doc_copy = doc.copy()
                    doc_copy["score"] = 0.5 * match_ratio  # Puntuación proporcional a cuántos términos coinciden
                    results.append(doc_copy)
                    logger.info(f"Found partial match in document: {doc.get('title')} (score: {doc_copy['score']:.2f})")
        
        # Ordenar por puntuación (de mayor a menor)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Text-based search found {len(results)} results")
        return results
