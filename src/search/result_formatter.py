"""
Module for formatting search results for presentation.
"""
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultFormatter:
    """Format search results for display to the user."""
    
    def __init__(self, 
                 max_snippet_length: int = 200, 
                 highlight_matches: bool = True,
                 show_scores: bool = False):
        """
        Initialize the result formatter.
        
        Args:
            max_snippet_length (int): Maximum length for content snippets
            highlight_matches (bool): Whether to highlight potential matches
            show_scores (bool): Whether to show similarity scores
        """
        self.highlight_matches = highlight_matches
        self.show_scores = show_scores
        logger.info(f"Initializing result formatter with max_snippet_length={max_snippet_length}")
        
    def format_results(self, results: List[Dict[str, Any]], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format search results for display.
        
        Args:
            results (List[Dict[str, Any]]): Raw search results
            query (str, optional): Original query for highlighting
            
        Returns:
            List[Dict[str, Any]]: Formatted results
        """
        formatted_results = []
        
        for i, result in enumerate(results):
            # Create a new formatted result
            formatted_result = {
                "position": i + 1,
                "title": result.get("title", "Untitled Document"),
                "url": result.get("url", "No URL available")
            }
            
            # Add score if enabled
            if self.show_scores and "score" in result:
                formatted_result["score"] = f"{result['score']:.4f}"
                
            formatted_results.append(formatted_result)
            
        logger.info(f"Formatted {len(formatted_results)} results")
        return formatted_results
        
    def display_results(self, results: List[Dict[str, Any]], query: Optional[str] = None) -> str:
        """
        Create a string representation of formatted results.
        
        Args:
            results (List[Dict[str, Any]]): Raw or formatted search results
            query (str, optional): Original query for context
            
        Returns:
            str: Formatted results as display text
        """
        # Format results if they haven't been formatted yet
        if results and "position" not in results[0]:
            results = self.format_results(results, query)
            
        if not results:
            return "No results found for your query."
            
        # Build the display text
        display_text = [f"Search Results{f' for: {query}' if query else ''}"]
        display_text.append("=" * 100)
        
        for result in results:
            display_text.append(f"[{result['position']}] {result['title']}")
            display_text.append(f"URL: {result['url']}")
            if self.show_scores and "score" in result:
                display_text.append(f"Relevance Score: {result['score']}")
            display_text.append("-" * 100)
            
        return "\n".join(display_text)