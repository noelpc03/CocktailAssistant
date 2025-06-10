"""
Search agent responsible for processing user queries and formatting results.
"""
import logging
import numpy as np
from typing import Dict, Any, List
import time

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class SearchAgent(Agent):
    """Agent responsible for search operations"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the search agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("search_agent")
        self.data_store = DataStore()
        self.max_results = self.config.get("max_results", 5)
        self.show_scores = self.config.get("show_scores", True)
        
    async def start(self) -> None:
        """Start the search agent"""
        await super().start()
        logger.info(f"Search agent {self.agent_id} started")
    
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
        
        if action == "process_query":
            query = content.get("query", "")
            
            if not query:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "search_results",
                        "status": "error",
                        "message": "Empty query"
                    }
                }
                
            # Process the query
            query_embedding = await self.process_query(query)
            
            # Request retrieval from the retrieval agent
            await self.send_message("retrieval_agent", {
                "action": "query_embeddings",
                "query_vector": query_embedding,
                "top_k": self.max_results,
                "original_sender": message["sender"],
                "query": query
            })
            
            # No direct response, will be sent by retrieval agent
            return None
            
        elif action == "query_results":
            # Results from retrieval agent
            results = content.get("results", [])
            original_sender = content.get("original_sender")
            query = content.get("query", "")
            status = content.get("status", "success")
            error_msg = content.get("message", "")
            
            if not original_sender:
                # No one to respond to
                return None
            
            # Check if there was an error
            if status == "error" or not results:
                if not error_msg:
                    error_msg = "No se encontraron resultados. Por favor, ejecuta primero el comando 'crawl'."
                
                logger.warning(f"Search error: {error_msg}")
                
                return {
                    "recipient": original_sender,
                    "content": {
                        "action": "search_results",
                        "status": status,
                        "query": query,
                        "message": error_msg,
                        "results": []
                    }
                }
                
            # Format the results
            formatted_results = self.format_results(results)
            
            return {
                "recipient": original_sender,
                "content": {
                    "action": "search_results",
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                    "result_count": len(results)
                }
            }
            
        return None
    
    async def process_query(self, query: str) -> np.ndarray:
        """
        Process and vectorize a user query.
        
        Args:
            query: The user's search query
            
        Returns:
            Query embedding vector
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        # Store the original query text in the data store for text-based search backup
        self.data_store.set("current_query", query)
        
        try:
            # Use the same model as in the working implementation
            from sentence_transformers import SentenceTransformer
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            
            # Load the model if not already loaded
            if not hasattr(self, 'model'):
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                
            # Encode the query
            embedding = self.model.encode(query)
            embedding = embedding / np.linalg.norm(embedding)
            
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.4f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Create a fallback embedding as a last resort
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        return embedding
    
    def format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results for presentation.
        
        Args:
            results: Raw search results
            
        Returns:
            Formatted search results
        """
        formatted_results = []
        
        for i, result in enumerate(results):
            formatted_result = {
                "rank": i + 1,
                "title": result.get("title", "Untitled"),
                "source_url": result.get("source_url", ""),
                "snippet": self._create_snippet(result.get("text", "")),
            }
            
            if self.show_scores and "score" in result:
                formatted_result["score"] = f"{result['score']:.4f}"
                
            formatted_results.append(formatted_result)
            
        return formatted_results
    
    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """
        Create a snippet from text.
        
        Args:
            text: Source text
            max_length: Maximum snippet length
            
        Returns:
            Text snippet
        """
        if len(text) <= max_length:
            return text
            
        # Truncate and add ellipsis
        return text[:max_length-3] + "..."
