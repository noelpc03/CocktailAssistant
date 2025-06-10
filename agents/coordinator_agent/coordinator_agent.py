"""
Coordinator agent responsible for orchestrating the multi-agent system.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
import os
import json

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore
from agents.common.message_broker import MessageBroker

logger = logging.getLogger(__name__)

class CoordinatorAgent(Agent):
    """Agent responsible for orchestrating other agents in the system"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the coordinator agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("coordinator_agent")
        self.data_store = DataStore()
        self.message_broker = MessageBroker()
        self.timeout = self.config.get("timeout", 30)
        
        # Track active operations
        self.active_operations = {}
        self.registered_agents = {}
        
        # Completion callbacks for operations
        self.completion_callbacks = {}
        
    async def start(self) -> None:
        """Start the coordinator agent"""
        await super().start()
        
        # Initialize message broker
        await self.message_broker.start()
        
        logger.info(f"Coordinator agent {self.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the coordinator agent"""
        await self.message_broker.stop()
        await super().stop()
        logger.info(f"Coordinator agent {self.agent_id} stopped")
    
    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """
        Register an agent with the coordinator.
        
        Args:
            agent_id: The ID of the agent
            agent: The agent instance
        """
        self.registered_agents[agent_id] = agent
        
        # Subscribe agent to message broker
        self.message_broker.subscribe(agent_id, agent.receive_message)
        
        logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the coordinator.
        
        Args:
            agent_id: The ID of the agent to unregister
        """
        if agent_id in self.registered_agents:
            # Unsubscribe from message broker
            self.message_broker.unsubscribe(agent_id)
            
            # Remove from registered agents
            del self.registered_agents[agent_id]
            
            logger.info(f"Unregistered agent: {agent_id}")
    
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
        
        if action == "search_query":
            # Process a user search query
            query = content.get("query", "")
            use_llm = content.get("use_llm", True)
            
            if not query:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "search_response",
                        "status": "error",
                        "message": "Empty query"
                    }
                }
                
            # Start search process
            operation_id = f"search_{hash(query)}_{id(query)}"
            self.active_operations[operation_id] = {
                "type": "search",
                "query": query,
                "use_llm": use_llm,
                "requester": message["sender"],
                "status": "in_progress",
                "results": None
            }
            
            # Send to search agent
            await self.send_message("search_agent", {
                "action": "process_query",
                "query": query,
                "operation_id": operation_id
            })
            
            # No immediate response
            return None
        
        elif action == "search_results":
            # Results from search agent
            query = content.get("query", "")
            results = content.get("results", [])
            
            # Find the operation
            operation = None
            operation_id = None
            for op_id, op in self.active_operations.items():
                if op["type"] == "search" and op["query"] == query:
                    operation = op
                    operation_id = op_id
                    break
            
            if not operation:
                # No matching operation found
                return None
                
            # Update operation
            operation["results"] = results
            
            # If LLM is requested, forward to generation agent
            if operation["use_llm"]:
                await self.send_message("generation_agent", {
                    "action": "generate_response",
                    "query": query,
                    "search_results": results,
                    "operation_id": operation_id
                })
                return None
            else:
                # Send a message indicating that LLM is required
                self.active_operations[operation_id]["status"] = "completed"
                
                return {
                    "recipient": operation["requester"],
                    "content": {
                        "action": "search_response",
                        "status": "success",
                        "query": query,
                        "results": [],
                        "generated_response": "Por favor, usa la opción de LLM para obtener una respuesta generada."
                    }
                }
        
        elif action == "generation_results":
            # Results from generation agent
            query = content.get("query", "")
            response = content.get("response", "")
            
            # Find the operation
            operation = None
            operation_id = None
            for op_id, op in self.active_operations.items():
                if op["type"] == "search" and op["query"] == query:
                    operation = op
                    operation_id = op_id
                    break
            
            if not operation:
                # No matching operation found
                return None
                
            # Update operation status
            self.active_operations[operation_id]["status"] = "completed"
            
            # Send only the response to requester, not the search results
            return {
                "recipient": operation["requester"],
                "content": {
                    "action": "search_response",
                    "status": "success",
                    "query": query,
                    "results": [],  # No enviamos los resultados de búsqueda
                    "generated_response": response
                }
            }
            
        elif action == "crawl_and_index":
            # Start the crawl and index process
            urls = content.get("urls")
            requester = message["sender"]
            
            # Generate operation ID
            operation_id = f"index_{id(urls)}_{hash(str(urls))}"
            
            logger.info(f"Starting crawl and index with operation_id: {operation_id}")
            
            # Track operation
            self.active_operations[operation_id] = {
                "type": "index",
                "urls": urls,
                "requester": requester,
                "status": "in_progress"
            }
            
            # Send to crawler agent to start the process
            await self.send_message("crawler_agent", {
                "action": "crawl_urls",
                "urls": urls,
                "operation_id": operation_id
            })
            
            # No immediate response
            return None
            
        elif action == "crawl_results":
            # Results from crawler agent - forward to vectorizer
            documents = content.get("documents", [])
            operation_id = content.get("operation_id")

            if not operation_id or operation_id not in self.active_operations:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return None

            operation = self.active_operations[operation_id]

            # Forward documents to vectorizer, incluyendo operation_id
            await self.send_message("vectorizer_agent", {
                "action": "vectorize_documents",
                "documents": documents,
                "operation_id": operation_id
            })

            # No immediate response
            return None

        elif action == "vectorize_results":
            # Results from vectorizer - forward to storage
            operation_id = content.get("operation_id")

            if not operation_id or operation_id not in self.active_operations:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return None

            operation = self.active_operations[operation_id]

            # Forward to retrieval agent for storage, incluyendo operation_id
            await self.send_message("retrieval_agent", {
                "action": "store_embeddings",
                "operation_id": operation_id
            })

            # No immediate response
            return None
            
        elif action == "store_results":
            # Results from storage
            operation_id = content.get("operation_id")
            status = content.get("status", "error")
            
            if not operation_id or operation_id not in self.active_operations:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return None
                
            operation = self.active_operations[operation_id]
            operation["status"] = "completed" if status == "success" else "failed"
            
            # Check if there's a completion callback for this operation
            operation_type = operation.get("type")
            if operation_type == "index" and "index" in self.completion_callbacks:
                for future in self.completion_callbacks["index"]:
                    if not future.done():
                        future.set_result({
                            "status": status,
                            "message": f"Indexing process {'completed successfully' if status == 'success' else 'failed'}"
                        })
            
            # Notify requester
            return {
                "recipient": operation["requester"],
                "content": {
                    "action": "index_response",
                    "status": status,
                    "message": f"Indexing process {'completed successfully' if status == 'success' else 'failed'}"
                }
            }
            
        return None
    
    async def start_crawl_and_index(self, urls: List[str] = None) -> bool:
        """
        Start the crawl and index process.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            Success flag
        """
        try:
            if not urls:
                # Use default URLs from config
                urls = self.config.get("default_urls", [])
            
            # Send message to start the process
            await self.process_message({
                "sender": self.agent_id,
                "recipient": self.agent_id,
                "content": {
                    "action": "crawl_and_index",
                    "urls": urls
                }
            })
            
            return True
        except Exception as e:
            logger.error(f"Error starting crawl and index: {e}")
            return False
    
    def register_crawl_completion_callback(self, future: asyncio.Future) -> None:
        """
        Register a callback for when crawling completes.
        
        Args:
            future: Future to be completed when crawling is done
        """
        logger.info(f"Registering crawl completion callback")
        
        if "index" not in self.completion_callbacks:
            self.completion_callbacks["index"] = []
        
        # Add the future to the list    
        self.completion_callbacks["index"].append(future)
        
        # Check if there are any completed operations already
        for op_id, op in self.active_operations.items():
            if op["type"] == "index" and op["status"] == "completed":
                logger.info(f"Found completed index operation {op_id}, completing future immediately")
                if not future.done():
                    future.set_result({
                        "status": "success",
                        "message": "Indexing process already completed"
                    })
                return
    
    async def search(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Perform a search query.
        
        Args:
            query: The search query
            use_llm: Whether to use LLM to enhance results
            
        Returns:
            Search results
        """
        try:
            # Create a future to wait for the response
            future = asyncio.Future()
            
            # Create a temporary message handler
            async def response_handler(message: Dict[str, Any]) -> None:
                content = message.get("content", {})
                if content.get("action") == "search_response" and content.get("query") == query:
                    if not future.done():
                        future.set_result(content)
            
            # Subscribe to responses
            temp_id = f"temp_{id(query)}"
            self.message_broker.subscribe(temp_id, response_handler)
            
            # Send the search request
            await self.process_message({
                "sender": temp_id,
                "recipient": self.agent_id,
                "content": {
                    "action": "search_query",
                    "query": query,
                    "use_llm": use_llm
                }
            })
            
            # Wait for response with timeout
            try:
                result = await asyncio.wait_for(future, self.timeout)
                return result
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "Search timed out"
                }
            finally:
                # Unsubscribe
                self.message_broker.unsubscribe(temp_id)
                
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
