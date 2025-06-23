#!/usr/bin/env python3
"""
Main script to run the multiagent bartender information retrieval system.
"""
import os
import sys
import asyncio
import logging
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from agents import (
    CrawlerAgent,
    VectorizerAgent, 
    RetrievalAgent,
    SearchAgent,
    GenerationAgent,
    OntologyAgent,
    StrategyAgent,
    CoordinatorAgent,
    MessageBroker,
    ConfigManager
)
from agents.dynamic_crawler_agent.dynamic_crawler_agent import DynamicCrawlerAgent

# Configure logging
log_level_str = os.environ.get("LOGLEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bartender_agent_system.log")
    ]
)
print(f"Logging level set to: {log_level_str}")
logger = logging.getLogger("agent_system")

class AgentSystem:
    """Main class for managing the multiagent system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the agent system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_manager = ConfigManager()
        self.coordinator = None
        self.running = False
        
        # Load token
        self.token_path = os.path.join(project_root, "tokenMixtral.txt")
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self) -> None:
        """Create all agents in the system"""
        logger.info("Creating agents")
        
        # Create coordinator first
        self.coordinator = CoordinatorAgent("coordinator_agent")
        
        # Create all other agents
        crawler = CrawlerAgent("crawler_agent")
        vectorizer = VectorizerAgent("vectorizer_agent")
        retriever = RetrievalAgent("retrieval_agent")
        searcher = SearchAgent("search_agent")
        generator = GenerationAgent("generation_agent")
        ontology = OntologyAgent("ontology_agent")
        strategy = StrategyAgent("strategy_agent")
        dynamic_crawler = DynamicCrawlerAgent("dynamic_crawler_agent")
        
        # Register all agents with coordinator
        self.coordinator.register_agent("crawler_agent", crawler)
        self.coordinator.register_agent("vectorizer_agent", vectorizer)
        self.coordinator.register_agent("retrieval_agent", retriever)
        self.coordinator.register_agent("search_agent", searcher)
        self.coordinator.register_agent("generation_agent", generator)
        self.coordinator.register_agent("strategy_agent", strategy)
        self.coordinator.register_agent("ontology_agent", ontology)
        self.coordinator.register_agent("dynamic_crawler_agent", dynamic_crawler)
        self.coordinator.register_agent("coordinator_agent", self.coordinator)
        
        logger.info("All agents created and registered")
    
    async def start(self) -> None:
        """Start the agent system"""
        if self.running:
            logger.warning("Agent system is already running")
            return
            
        logger.info("Starting agent system")
        
        # Start all agents through coordinator
        # El coordinador ya maneja el inicio de MessageBroker
        await self.coordinator.start()
        
        # Iniciar manualmente el agente de estrategia para asegurar que cargue su API key
        if "strategy_agent" in self.coordinator.registered_agents:
            strategy_agent = self.coordinator.registered_agents["strategy_agent"]
            await strategy_agent.start()
            logger.debug("Strategy agent started explicitly to ensure API key loading")
        
        self.running = True
        logger.info("Agent system started")
    
    async def stop(self) -> None:
        """Stop the agent system"""
        if not self.running:
            logger.warning("Agent system is not running")
            return
            
        logger.info("Stopping agent system")
        
        # Stop coordinator (which will handle stopping other agents)
        await self.coordinator.stop()
        
        self.running = False
        logger.info("Agent system stopped")
    
    async def crawl_and_index(self, urls: List[str] = None, wait_for_completion: bool = False) -> Dict[str, Any]:
        """
        Start the crawl and index process.
        
        Args:
            urls: List of URLs to crawl
            wait_for_completion: Whether to wait for the crawl process to complete
            
        Returns:
            Status information
        """
        if not self.running:
            await self.start()
            
        logger.info("Starting crawl and index process")
        result = await self.coordinator.start_crawl_and_index(urls)
        
        if wait_for_completion:
            # Create a future that will be completed when crawling is done
            crawl_future = asyncio.Future()
            self.coordinator.register_crawl_completion_callback(crawl_future)
            
            try:
                # Wait for the crawl to complete with a timeout
                await asyncio.wait_for(crawl_future, timeout=7200)  # 120-minute timeout (2 horas)
                return {
                    "success": True,
                    "message": "Crawl and index process completed"
                }
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "message": "Crawl and index process timed out"
                }
        
        return {
            "success": result,
            "message": "Crawl and index process started"
        }
    
    async def search(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Perform a search query.
        
        Args:
            query: The search query
            use_llm: Whether to use LLM to enhance results
            
        Returns:
            Search results
        """
        if not self.running:
            await self.start()
            
        logger.info(f"Searching for: {query} (use_llm={use_llm})")
        results = await self.coordinator.search(query, use_llm)
        
        return results
        
    async def extract_ontology(self) -> Dict[str, Any]:
        """
        Extract ontology from indexed documents.
        
        Returns:
            Status information
        """
        if not self.running:
            await self.start()
            
        logger.info("Starting ontology extraction")
        results = await self.coordinator.extract_ontology()
        
        return results
    
    async def query_ontology(self, query: str, use_natural_language: bool = True) -> Dict[str, Any]:
        """
        Query the ontology.
        
        Args:
            query: SPARQL query or natural language query
            use_natural_language: Whether the query is in natural language
            
        Returns:
            Query results
        """
        if not self.running:
            await self.start()
            
        logger.info(f"Querying ontology: {query}")
        results = await self.coordinator.query_ontology(query, use_natural_language)
        
        return results
        
    async def visualize_ontology(self) -> Dict[str, Any]:
        """
        Generate visualization for the ontology.
        
        Returns:
            Status information with path to visualization file
        """
        if not self.running:
            await self.start()
            
        logger.info("Generating ontology visualization")
        results = await self.coordinator.visualize_ontology()
        
        return results


async def run_cli():
    """Run the system with command line interface"""
    parser = argparse.ArgumentParser(description="Bartender Information Retrieval System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl and index information")
    crawl_parser.add_argument("--urls", nargs="+", help="URLs to crawl")
    crawl_parser.add_argument("--wait", action="store_true", help="Wait for crawl to complete before exiting")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for information")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--no-llm", action="store_true", help="Don't use LLM")
    
    # Ontology commands
    ontology_parser = subparsers.add_parser("ontology", help="Ontology operations")
    ontology_subparsers = ontology_parser.add_subparsers(dest="ontology_command", help="Ontology command to run")
    
    # Extract ontology command
    extract_parser = ontology_subparsers.add_parser("extract", help="Extract ontology from indexed documents")
    
    # Query ontology command
    query_parser = ontology_subparsers.add_parser("query", help="Query the ontology")
    query_parser.add_argument("query_text", help="Query text (SPARQL or natural language)")
    query_parser.add_argument("--sparql", action="store_true", help="Query is in SPARQL format")
    
    # Visualize ontology command
    visualize_parser = ontology_subparsers.add_parser("visualize", help="Generate visualization of the ontology")
    
    args = parser.parse_args()
    
    # Initialize system
    system = AgentSystem()
    await system.start()
    
    try:
        if args.command == "crawl":
            if args.wait:
                print("Starting crawl and waiting for completion...")
                result = await system.crawl_and_index(args.urls, wait_for_completion=True)
                print(f"Crawl result: {result['message']}")
            else:
                result = await system.crawl_and_index(args.urls)
                print(f"Crawl started: {result['message']}")
                print("This is an asynchronous operation. Check logs for progress.")
                print("Use --wait flag to wait for completion before exiting.")
            
        elif args.command == "search":
            result = await system.search(args.query, not args.no_llm)
            
            if result.get("status") == "success":
                print(f"Results for: {args.query}")
                print("-" * 50)
                
                if result.get("generated_response"):
                    print(result["generated_response"])
                    print("-" * 50)  
                else:
                    print("No se pudo generar una respuesta.")
                    print("-" * 50)  
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
                
        elif args.command == "ontology":
            if args.ontology_command == "extract":
                print("Extracting ontology from indexed documents...")
                result = await system.extract_ontology()
                
                if result.get("status") == "success":
                    print(f"Ontology extraction successful: {result.get('message')}")
                    print(f"Ontology saved to: {result.get('ontology_file')}")
                else:
                    print(f"Error: {result.get('message', 'Unknown error')}")
                    
            elif args.ontology_command == "query":
                use_natural_language = not args.sparql
                result = await system.query_ontology(args.query_text, use_natural_language)
                
                print("\n\n")  # Add some clear space
                if result.get("status") == "success":
                    print(f"=== QUERY RESULTS ===")
                    print("=" * 50)
                    
                    # Print results in tabular format
                    results = result.get("results", [])
                    if results:
                        # First, check if there's a natural language response (for natural language queries)
                        nl_responses = [r.get("nl_response") for r in results if "nl_response" in r]
                        if nl_responses and nl_responses[0] and use_natural_language:
                            print("RESPUESTA EN LENGUAJE NATURAL:")
                            print("-" * 50)
                            print(nl_responses[0])
                            print("-" * 50)
                            print("\nDETALLES TÉCNICOS:")
                            print("-" * 50)
                            
                        # Extract headers from results that aren't metadata
                        valid_results = [r for r in results if not any(key in r for key in ["generated_sparql", "info", "nl_response"])]
                        
                        if valid_results:
                            # Extract headers
                            headers = list(valid_results[0].keys())
                            
                            # Print headers
                            header_str = " | ".join(headers)
                            print(header_str)
                            print("-" * len(header_str))
                            
                            # Print rows
                            for row in valid_results:
                                values = [str(row.get(h, "")) for h in headers]
                                print(" | ".join(values))
                                
                            print(f"\nFound {len(valid_results)} cocktails matching the query.")
                        else:
                            # Look for info messages if no valid results and no NL response shown
                            if not (nl_responses and nl_responses[0] and use_natural_language):
                                info_messages = [r.get("info") for r in results if "info" in r]
                                if info_messages:
                                    print("\n".join(info_messages))
                                else:
                                    print("No results found")
                    else:
                        print("No results found")
                else:
                    print(f"Error: {result.get('message', 'Unknown error')}")
                
                # Flush stdout to ensure output is displayed
                import sys
                sys.stdout.flush()
                    
            elif args.ontology_command == "visualize":
                print("Generating ontology visualization...")
                result = await system.visualize_ontology()
                
                if result.get("status") == "success":
                    print(f"Visualization generated: {result.get('visualization_file')}")
                else:
                    print(f"Error: {result.get('message', 'Unknown error')}")
            else:
                print("No ontology command specified. Use 'extract', 'query', or 'visualize'.")
        else:
            print("No command specified. Use --help for usage information.")
            
    finally:
        await system.stop()


def main():
    """Main entry point"""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
