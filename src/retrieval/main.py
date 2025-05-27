"""
Main module to orchestrate the bartender information retrieval process.
"""
import os
import logging
import json
from typing import List, Dict, Any

from src.retrieval.crawler.web_crawler import scrape_urls, get_all_bartender_content
from src.retrieval.embeddings.vectorizer import vectorize_documents
from src.retrieval.storage.faiss_store import store_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bartender_retrieval.log")
    ]
)
logger = logging.getLogger(__name__)

# Default URLs for bartender information
DEFAULT_URLS = [
    "https://www.liquor.com/recipes/margarita/",
    "https://www.diffordsguide.com/cocktails/recipe/42/martini",
    "https://www.thespruceeats.com/classic-cocktails-everyone-should-know-760778",
    "https://en.wikipedia.org/wiki/Bartender",
]

# Default paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

def ensure_directories():
    """Ensure all necessary directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def save_metadata(documents: List[Dict[str, Any]], path: str = METADATA_PATH):
    """Save document metadata to a JSON file."""
    with open(path, "w") as f:
        json.dump(documents, f, indent=2)
    logger.info(f"Saved metadata for {len(documents)} documents to {path}")

def retrieve_and_process(urls: List[str] = None, save_dir: str = EMBEDDINGS_DIR):
    """
    Retrieve information from URLs, vectorize it, and store it.
    
    Args:
        urls (List[str], optional): List of URLs to process. If None, use all predefined URLs
        save_dir (str, optional): Directory to save the embeddings
        
    Returns:
        dict: Statistics about the processing
    """
    # Get all content if no specific URLs provided
    if urls is None:
        logger.info("Using predefined bartender URLs")
        documents = get_all_bartender_content()
    else:
        logger.info(f"Processing {len(urls)} provided URLs")
        documents = scrape_urls(urls)
    
    if not documents:
        logger.error("No documents were retrieved. Aborting.")
        return {"status": "error", "message": "No documents retrieved"}
        
    # Save raw metadata
    ensure_directories()
    save_metadata(documents)
    
    # Vectorize documents
    embeddings = vectorize_documents(documents)
    
    # Store embeddings
    storage = store_embeddings(embeddings, documents, save_dir)
    
    return {
        "status": "success",
        "document_count": len(documents),
        "embedding_dimension": embeddings.shape[1],
        "storage_location": save_dir
    }

if __name__ == "__main__":
    logger.info("Starting bartender information retrieval process")
    
    # Process default URLs
    result = retrieve_and_process(DEFAULT_URLS)
    
    if result["status"] == "success":
        logger.info(f"Successfully processed {result['document_count']} documents")
        logger.info(f"Embeddings saved to {result['storage_location']}")
    else:
        logger.error(f"Processing failed: {result['message']}")
