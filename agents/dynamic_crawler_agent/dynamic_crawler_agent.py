"""
Dynamic crawler agent for fetching and processing web content in real-time.
Provides information from web searches when ontology or embeddings are insufficient.
"""
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import json

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore
from agents.dynamic_crawler.dynamic_crawler import DynamicCrawler

logger = logging.getLogger(__name__)

class DynamicCrawlerAgent(Agent):
    """
    Agent responsible for dynamic web crawling to supplement ontology and embeddings search.
    Retrieves real-time information from the web when needed.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the dynamic crawler agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("dynamic_crawler_agent")
        self.data_store = DataStore()
        
        # Initialize the dynamic crawler utility
        self.dynamic_crawler = DynamicCrawler()
        
        # Cache for recent searches to avoid duplicate work
        self.search_cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # Cache time-to-live in seconds
        self.max_cache_size = self.config.get("max_cache_size", 100)
        
        # Threshold for considering search results sufficient
        self.min_results_threshold = self.config.get("min_results_threshold", 3)
    
    async def start(self) -> None:
        """Start the dynamic crawler agent"""
        await super().start()
        logger.info(f"Dynamic crawler agent {self.agent_id} started")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        content = message.get("content", {})
        action = content.get("action", "")
        
        if action == "search_web":
            # Proactive search from strategy agent or reactive search from coordinator
            return await self._handle_search_web(content, message)
        
        elif action == "complement_results":
            # Complementary search when embeddings or ontology results aren't sufficient
            return await self._handle_complement_results(content, message)
        
        else:
            logger.warning(f"Unknown action '{action}' received by dynamic crawler agent")
            return None
    
    async def _handle_search_web(self, content: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request to search the web for information.
        
        Args:
            content: Message content
            message: Full message
            
        Returns:
            Response message
        """
        query = content.get("query", "")
        operation_id = content.get("operation_id", "")
        source = content.get("source", "unknown")  # Where the request came from
        callback_agent = content.get("callback_agent", "coordinator_agent")  # Por defecto enviar al coordinador
        
        if not query:
            return self._error_response(message["sender"], "Empty query", operation_id)
        
        logger.info(f"Searching web for query: '{query}' (operation_id: {operation_id}, source: {source})")
        
        try:
            # Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.search_cache:
                logger.info(f"Using cached results for query: '{query}'")
                web_info = self.search_cache[cache_key]["info"]
            else:
                # Execute search in thread to not block
                search_results = await asyncio.to_thread(
                    self.dynamic_crawler.search_web, 
                    query, 
                    self.config.get("num_search_results", 5)
                )
                
                if not search_results or len(search_results) < self.min_results_threshold:
                    logger.warning(f"Insufficient search results for query: '{query}'")
                    return self._error_response(
                        message["sender"], 
                        f"Insufficient search results ({len(search_results) if search_results else 0} found)", 
                        operation_id
                    )
                
                # Extract relevant information
                web_info = await asyncio.to_thread(
                    self.dynamic_crawler.extract_relevant_info,
                    query,
                    search_results
                )
                
                # Cache the results
                self._update_cache(cache_key, web_info)
            
            # Return the results directly to the sender
            return {
                "recipient": "coordinator_agent",  # Siempre enviar al coordinador
                "content": {
                    "action": "web_search_results",
                    "status": "success",
                    "query": query,
                    "web_info": web_info,
                    "operation_id": operation_id,
                    "source": source,
                    "original_sender": message["sender"]  # Guardamos quién hizo la solicitud original
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}")
            return self._error_response("coordinator_agent", f"Error searching web: {str(e)}", operation_id)
    
    async def _handle_complement_results(self, content: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a request to complement existing results with web information.
        
        Args:
            content: Message content
            message: Full message
            
        Returns:
            Response message
        """
        query = content.get("query", "")
        operation_id = content.get("operation_id", "")
        existing_results = content.get("existing_results", [])
        ontology_results = content.get("ontology_results", [])
        
        if not query:
            return self._error_response(message["sender"], "Empty query", operation_id)
        
        context_type = "ontology" if ontology_results else "embeddings"
        logger.info(f"Complementing {context_type} results for query: '{query}' (operation_id: {operation_id})")
        
        try:
            # Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.search_cache:
                logger.info(f"Using cached results to complement {context_type} for query: '{query}'")
                web_info = self.search_cache[cache_key]["info"]
            else:
                # Execute search in thread to not block
                search_results = await asyncio.to_thread(
                    self.dynamic_crawler.search_web, 
                    query, 
                    self.config.get("num_search_results", 5)
                )
                
                if not search_results:
                    logger.warning(f"No search results found to complement {context_type} for query: '{query}'")
                    return self._error_response(
                        message["sender"], 
                        f"No web results found to complement {context_type}",
                        operation_id
                    )
                
                # Extract relevant information
                web_info = await asyncio.to_thread(
                    self.dynamic_crawler.extract_relevant_info,
                    query,
                    search_results
                )
                
                # Cache the results
                self._update_cache(cache_key, web_info)
            
            # Return the web information along with the existing results
            return {
                "recipient": "coordinator_agent",  # Siempre enviar al coordinador
                "content": {
                    "action": "complement_results_response",
                    "status": "success",
                    "query": query,
                    "operation_id": operation_id,
                    "web_info": web_info,
                    "existing_results": existing_results,
                    "ontology_results": ontology_results,
                    "context_type": context_type,
                    "original_sender": message["sender"]  # Guardamos quién hizo la solicitud original
                }
            }
            
        except Exception as e:
            logger.error(f"Error complementing {context_type} results: {str(e)}")
            return self._error_response("coordinator_agent", f"Error complementing results: {str(e)}", operation_id)
    
    def _update_cache(self, key: str, web_info: str) -> None:
        """Update the search cache with new results"""
        import time
        
        # Add new entry
        self.search_cache[key] = {
            "info": web_info,
            "timestamp": time.time()
        }
        
        # Clean old entries if cache is too big
        if len(self.search_cache) > self.max_cache_size:
            # Remove oldest entries
            now = time.time()
            expired_keys = [
                k for k, v in self.search_cache.items() 
                if now - v["timestamp"] > self.cache_ttl
            ]
            
            for k in expired_keys:
                del self.search_cache[k]
            
            # If still too big, remove oldest by timestamp
            if len(self.search_cache) > self.max_cache_size:
                sorted_keys = sorted(
                    self.search_cache.keys(),
                    key=lambda k: self.search_cache[k]["timestamp"]
                )
                
                to_remove = len(self.search_cache) - self.max_cache_size
                for k in sorted_keys[:to_remove]:
                    del self.search_cache[k]
    
    def _error_response(self, recipient: str, error_message: str, operation_id: str = "") -> Dict[str, Any]:
        """Generate an error response"""
        return {
            "recipient": recipient,
            "content": {
                "action": "web_search_error",
                "status": "error",
                "message": error_message,
                "operation_id": operation_id
            }
        }
