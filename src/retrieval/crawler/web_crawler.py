"""
Web crawler module for bartender information retrieval.
Handles fetching and extracting content from bartender-related websites.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of bartender-related websites to crawl
BARTENDER_URLS = [
    "https://www.liquor.com/recipes/",
    "https://www.thespruceeats.com/bartending-basics-4162637",
    "https://www.diffordsguide.com/cocktails",
    "https://en.wikipedia.org/wiki/Bartender",
    "https://www.masterclass.com/articles/bartending-guide",
]

def scrape_url(url: str) -> str:
    """
    Scrape content from a single URL.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        str: Extracted text content
    """
    try:
        logger.info(f"Scraping URL: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        content = '\n'.join(p.text.strip() for p in paragraphs if p.text.strip())
        
        # Extract headers for additional context
        headers = soup.find_all(['h1', 'h2', 'h3'])
        header_content = '\n'.join(h.text.strip() for h in headers if h.text.strip())
        
        # Combine content
        full_content = header_content + '\n\n' + content
        
        logger.info(f"Successfully scraped {len(full_content)} characters from {url}")
        return full_content
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""

def scrape_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Scrape content from multiple URLs.
    
    Args:
        urls (List[str]): List of URLs to scrape
        
    Returns:
        List[Dict[str, Any]]: List of documents with their metadata
    """
    documents = []
    
    for url in urls:
        content = scrape_url(url)
        if content:
            documents.append({
                "url": url,
                "text": content,
                "source": url.split('/')[2]  # Extract domain as source
            })
    
    logger.info(f"Scraped {len(documents)} documents successfully")
    return documents

def get_all_bartender_content() -> List[Dict[str, Any]]:
    """
    Get content from all predefined bartender URLs.
    
    Returns:
        List[Dict[str, Any]]: List of documents with their metadata
    """
    return scrape_urls(BARTENDER_URLS)

if __name__ == "__main__":
    # For testing purposes
    docs = get_all_bartender_content()
    print(f"Retrieved {len(docs)} documents")
