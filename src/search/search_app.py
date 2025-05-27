"""
Main search application for bartender information retrieval.
"""
import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Add the necessary directories to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src directory
project_root = os.path.dirname(parent_dir)  # project root
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)

# Define directories
EMBEDDINGS_DIR = os.path.join(parent_dir, "data", "embeddings")
TOKEN_PATH = os.path.join(project_root, "tokenGemini.txt")

# Import the search components using direct relative imports
from search.query_processor import QueryProcessor
from search.retrieval_engine import RetrievalEngine
from search.result_formatter import ResultFormatter

# Import the generation components
from generation.llm_generator import GeminiGenerator
from generation.prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bartender_search.log")
    ]
)
logger = logging.getLogger(__name__)

class BartenderSearchApp:
    """Main search application for bartender information."""
    
    def __init__(self, 
                 embeddings_dir: str = EMBEDDINGS_DIR,
                 token_path: str = TOKEN_PATH,
                 show_scores: bool = True,
                 max_results: int = 5,
                 use_llm: bool = True):
        """
        Initialize the search application components.
        
        Args:
            embeddings_dir (str): Path to embeddings directory
            token_path (str): Path to Gemini API token
            show_scores (bool): Whether to display similarity scores
            max_results (int): Maximum results to display
            use_llm (bool): Whether to use the LLM for response generation
        """
        logger.info(f"Initializing search application with embeddings from {embeddings_dir}")
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.retrieval_engine = RetrievalEngine(embeddings_dir)
        self.result_formatter = ResultFormatter(show_scores=show_scores)
        self.max_results = max_results
        self.use_llm = use_llm
        
        # Initialize LLM components if needed
        if use_llm:
            try:
                self.prompt_builder = PromptBuilder()
                self.llm_generator = GeminiGenerator(api_key_path=token_path, prompt_builder=self.prompt_builder)
                logger.info("LLM generator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM generator: {str(e)}")
                self.use_llm = False
        
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search with the given query.
        
        Args:
            query (str): Search query string
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        try:
            logger.info(f"Processing search query: '{query}'")
            
            # Convert query to embedding
            query_embedding = self.query_processor.vectorize_query(query)
            
            # Search for similar documents
            results = self.retrieval_engine.search(query_embedding, top_k=self.max_results)
            
            return results
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
            
    def generate_llm_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM based on the retrieved documents.
        
        Args:
            query (str): User query
            results (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            str: Generated response
        """
        if not self.use_llm:
            return "La generación con LLM está desactivada o no se pudo inicializar."
            
        try:
            # Generate response using the LLM
            response = self.llm_generator.generate(query, results)
            return response
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error al generar respuesta: {str(e)}"
    
    def display_search_results(self, query: str) -> str:
        """
        Search and display formatted results.
        
        Args:
            query (str): Search query
            
        Returns:
            str: Formatted results as display text
        """
        # Perform the search
        try:
            results = self.search(query)
        except Exception as e:
            logger.error(f"Error durante la búsqueda: {str(e)}")
            return f"Error al realizar la búsqueda: {str(e)}"
        
        # Format the search results
        display_text = self.result_formatter.display_results(results, query)
        
        # Generate LLM response if enabled and results found
        llm_response = None
        if self.use_llm and results:
            try:
                logger.info("Generando respuesta LLM para los resultados...")
                llm_response = self.generate_llm_response(query, results)
                
                # Append LLM response if available
                if llm_response:
                    display_text += "\n\n" + "=" * 100 + "\n"
                    display_text += "Respuesta generada por IA:\n"
                    display_text += "-" * 100 + "\n"
                    display_text += llm_response
            except Exception as e:
                logger.error(f"Error al generar respuesta LLM: {str(e)}")
                display_text += "\n\n" + "=" * 100 + "\n"
                display_text += "No se pudo generar una respuesta de IA debido a un error.\n"
                display_text += f"Error: {str(e)}"
        
        return display_text
        
def main():
    """Entry point for command-line search application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Bartender Information Search")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("-r", "--results", type=int, default=5, help="Maximum number of results")
    parser.add_argument("-s", "--scores", action="store_true", help="Show similarity scores")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM response generation")
    args = parser.parse_args()
    
    # Create the search application
    search_app = BartenderSearchApp(
        show_scores=args.scores,
        max_results=args.results,
        use_llm=not args.no_llm
    )
    
    # Get query from command-line arguments or prompt
    query = args.query
    if not query:
        query = input("Enter your search query: ")
    
    # Display search results
    print()
    print(search_app.display_search_results(query))
    
if __name__ == "__main__":
    main()