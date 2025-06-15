"""
Coordinator agent responsible for orchestrating the multi-agent system.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
import os
import json

from agents.common.agent_interface import Agent
from agents.coordinator_agent.handle_ontology_results import handle_ontology_results, format_ontology_results_for_llm
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
        self.timeout = self.config.get("timeout", 120)  # Increased timeout to 120 seconds
        
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
            
            # First send to strategy agent to determine search approach
            logger.info(f"Sending query '{query}' to strategy agent")
            await self.send_message("strategy_agent", {
                "action": "determine_strategy",
                "query": query,
                "operation_id": operation_id
            })
            
            # No immediate response
            return None
        
        elif action == "strategy_response":
            # Response from strategy agent
            query = content.get("query", "")
            strategy = content.get("strategy", "embedding")  # Default to embedding if not specified
            explanation = content.get("explanation", "")
            operation_id = content.get("operation_id", "")
            
            logger.info(f"Strategy determined for query '{query}': {strategy} - {explanation}")
            
            if not operation_id or operation_id not in self.active_operations:
                logger.warning(f"Invalid operation ID received from strategy agent: {operation_id}")
                return None
                
            # Update operation with strategy info
            self.active_operations[operation_id]["strategy"] = strategy
            self.active_operations[operation_id]["strategy_explanation"] = explanation
            
            # Depending on the strategy, send to appropriate agent
            if strategy == "ontology":
                logger.info(f"Using ontology-based search for query: {query}")
                await self.send_message("ontology_agent", {
                    "action": "query_ontology",
                    "query": query,
                    "operation_id": operation_id
                })
            else:  # Default to embedding search
                logger.info(f"Using embedding-based search for query: {query}")
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
            
        elif action == "ontology_results":
            # Handle the results from ontology agent
            # Using a dedicated method for better organization
            return await self._handle_ontology_results(content, message)
            
            if not operation:
                # No matching operation found
                logger.warning(f"No operation found for ontology results with ID {operation_id}")
                return None
                
            # Update operation status
            self.active_operations[operation_id]["status"] = "completed"
            self.active_operations[operation_id]["results"] = results
            
            # Check if use_llm is True and results exist
            use_llm = operation.get("use_llm", False)
            
            if use_llm and results:
                # Format results for LLM processing
                context = []
                
                for result in results:
                    result_type = result.get("type", "")
                    
                    if result_type == "cocktail_with_ingredient":
                        context.append(f"Cocktail {result['cocktail']} contains {result['ingredient']}.")
                    elif result_type == "ingredient_in_cocktail":
                        context.append(f"{result['cocktail']} contains {result['ingredient']}.")
                    elif result_type == "glass_for_cocktail":
                        context.append(f"{result['cocktail']} is served in a {result['glass']}.")
                    elif result_type == "method_for_cocktail":
                        context.append(f"{result['cocktail']} is prepared by {result['method']}.")
                    elif result_type == "cocktail_description":
                        context.append(f"{result['cocktail']}: {result['description']}")
                    elif result_type == "general_match":
                        context.append(f"{result['subject']} {result['predicate']} {result['object']}.")
                
                context_text = "\n".join(context)
                
                # Send to generation agent
                await self.send_message("generation_agent", {
                    "action": "generate_response",
                    "query": query,
                    "context": context_text,
                    "operation_id": operation_id
                })
                
                # No response yet, wait for generation
                return None
            else:
                # Send results directly to requester
                return {
                    "recipient": operation["requester"],
                    "content": {
                        "action": "search_response",
                        "status": "success", 
                        "query": query,
                        "results": results
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
    
    async def extract_ontology(self) -> Dict[str, Any]:
        """
        Extract ontology from indexed documents.
        
        Returns:
            Status information
        """
        try:
            # Create a future to wait for the response
            future = asyncio.Future()
            
            # Create a temporary message handler
            async def response_handler(message: Dict[str, Any]) -> None:
                print(f"[Coordinator] Handler de respuesta recibió mensaje: {message}")
                logger.info(f"Response handler received message: {message}")
                
                content = message.get("content", {})
                if content.get("action") == "ontology_results":
                    print(f"[Coordinator] Recibida respuesta de ontología")
                    logger.info(f"Received ontology_results action, completing future")
                    if not future.done():
                        future.set_result(content)
            
            # Subscribe to responses
            temp_id = f"temp_ontology_{id(future)}"
            self.message_broker.subscribe(temp_id, response_handler)
            
            # Send the ontology extraction request to ontology_agent
            logger.info(f"Sending extract_ontology request to ontology_agent from {temp_id}")
            print(f"[Coordinator] Enviando solicitud de extracción de ontología al agente de ontología")
            
            # Enviar mensaje asegurando que se incluyan sender y recipient correctamente
            message = {
                "sender": temp_id,
                "recipient": "ontology_agent",
                "content": {
                    "action": "extract_ontology",
                    "operation_id": temp_id
                }
            }
            
            print(f"[Coordinator] Mensaje completo a enviar: {message}")
            await self.message_broker.publish_message(message)
            print(f"[Coordinator] Mensaje publicado, esperando respuesta...")
            
            # Wait for response with timeout (mucho más tiempo para extracción de ontología)
            try:
                result = await asyncio.wait_for(future, self.timeout * 60)  # 60 veces el timeout normal para dar mucho más tiempo
                print(f"[Coordinator] Respuesta recibida: {result}")
                return result
            except asyncio.TimeoutError:
                print(f"[Coordinator] ❌ Timeout esperando respuesta de ontología")
                return {
                    "status": "error",
                    "message": "Ontology extraction timed out después de un tiempo extendido"
                }
            finally:
                # Unsubscribe
                self.message_broker.unsubscribe(temp_id)
                print(f"[Coordinator] Desuscrito de {temp_id}")
                
        except Exception as e:
            logger.error(f"Error extracting ontology: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def query_ontology(self, query: str, use_natural_language: bool = True) -> Dict[str, Any]:
        """
        Query the ontology with improved timeout handling.
        
        Args:
            query: SPARQL query or natural language query
            use_natural_language: Whether the query is in natural language
            
        Returns:
            Query results
        """
        try:
            # Log the start of the process
            print(f"[Coordinator] Iniciando consulta de ontología: '{query}' (lenguaje natural: {use_natural_language})")
            logger.info(f"Starting ontology query: '{query}' (natural language: {use_natural_language})")
            
            # Create a future to wait for the response
            future = asyncio.Future()
            
            # Create a temporary message handler
            async def response_handler(message: Dict[str, Any]) -> None:
                content = message.get("content", {})
                if content.get("action") == "query_results":
                    print(f"[Coordinator] Recibida respuesta de consulta de ontología")
                    if not future.done():
                        future.set_result(content)
            
            # Subscribe to responses
            temp_id = f"temp_query_{id(future)}"
            self.message_broker.subscribe(temp_id, response_handler)
            
            # Prepare message to send to ontology agent
            message = {
                "sender": temp_id,
                "recipient": "ontology_agent",
                "content": {
                    "action": "query_ontology",
                    "query": query,
                    "operation_id": temp_id,
                    "use_natural_language": use_natural_language
                }
            }
            
            # Directly send message to ontology agent
            print(f"[Coordinator] Enviando solicitud al agente de ontología...")
            await self.message_broker.publish_message(message)
            
            # Wait for response with extended timeout for ontology queries (90 seconds)
            # Reducido de 120 a 90 segundos ya que hemos implementado timeouts internos
            ontology_query_timeout = 90
            
            try:
                print(f"[Coordinator] Esperando respuesta (timeout: {ontology_query_timeout}s)...")
                result = await asyncio.wait_for(future, ontology_query_timeout)
                print(f"[Coordinator] ✓ Respuesta recibida: {len(str(result))} caracteres")
                return result
            except asyncio.TimeoutError:
                error_msg = f"La consulta de ontología excedió el tiempo límite de {ontology_query_timeout} segundos."
                logger.error(f"Ontology query timed out after {ontology_query_timeout} seconds")
                print(f"[Coordinator] ❌ {error_msg}")
                
                if use_natural_language:
                    error_msg += " Intente con una consulta más específica o use SPARQL directamente con --sparql."
                
                return {
                    "status": "error",
                    "message": error_msg
                }
            finally:
                # Unsubscribe to clean up
                self.message_broker.unsubscribe(temp_id)
                print(f"[Coordinator] Desuscrito de {temp_id}")
                
        except Exception as e:
            error_msg = f"Error al consultar la ontología: {str(e)}"
            logger.error(error_msg)
            print(f"[Coordinator] ❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
    
    async def visualize_ontology(self) -> Dict[str, Any]:
        """
        Generate visualization for the ontology.
        
        Returns:
            Status information with path to visualization file
        """
        try:
            # Create a future to wait for the response
            future = asyncio.Future()
            
            # Create a temporary message handler
            async def response_handler(message: Dict[str, Any]) -> None:
                content = message.get("content", {})
                if content.get("action") == "visualization_results":
                    if not future.done():
                        future.set_result(content)
            
            # Subscribe to responses
            temp_id = f"temp_visualization_{id(future)}"
            self.message_broker.subscribe(temp_id, response_handler)
            
            # Send the visualization request
            await self.process_message({
                "sender": temp_id,
                "recipient": self.agent_id,
                "content": {
                    "action": "visualize_ontology"
                }
            })
            
            # Wait for response with timeout
            try:
                result = await asyncio.wait_for(future, self.timeout)
                return result
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "message": "Ontology visualization timed out"
                }
            finally:
                # Unsubscribe
                self.message_broker.unsubscribe(temp_id)
                
        except Exception as e:
            logger.error(f"Error visualizing ontology: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def _handle_ontology_results(self, content, message):
        """
        Handle results from the ontology agent.
        
        Args:
            content: The message content
            message: The full message
            
        Returns:
            Response message or None
        """
        return await handle_ontology_results(self, content, message)
