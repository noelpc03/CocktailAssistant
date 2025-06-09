"""
Crawler agent responsible for fetching web content.
"""
import asyncio
import logging
import requests
from typing import Dict, Any, List
from bs4 import BeautifulSoup

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class CrawlerAgent(Agent):
    """Agent responsible for crawling web content"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the crawler agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("crawler_agent")
        self.data_store = DataStore()
    
    async def start(self) -> None:
        """Start the crawler agent"""
        await super().start()
        logger.info(f"Crawler agent {self.agent_id} started")
    
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
        
        if action == "crawl_urls":
            urls = content.get("urls", self.config.get("default_urls", []))
            operation_id = content.get("operation_id")
            
            logger.info(f"Received crawl_urls request with operation_id: {operation_id}")
            
            response_data = await self.crawl_urls(urls)
            
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "crawl_results",
                    "documents": response_data,
                    "operation_id": operation_id,
                    "status": "success"
                }
            }
        
        return None
    
    async def crawl_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs and extract content.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of documents with extracted content
        """
        documents = []
        
        for url in urls:
            try:
                logger.info(f"Crawling URL: {url}")
                content = await self._fetch_page(url)
                
                if content:
                    doc = {
                        "url": url,
                        "title": self._extract_title(content),
                        "content": self._clean_content(content),
                        "source": "web_crawl"
                    }
                    documents.append(doc)
                    logger.info(f"Successfully crawled {url}")
                else:
                    logger.warning(f"Failed to extract content from {url}")
                    
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        
        # Store the crawled documents
        self.data_store.set("crawled_documents", documents)
        
        return documents
    
    async def _fetch_page(self, url: str) -> str:
        """
        Fetch page content from URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content of the page
        """
        try:
            # Use a proper user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Run the HTTP request in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, headers=headers, timeout=30)
            )
            
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def _extract_title(self, html_content: str) -> str:
        """
        Extract title from HTML content.
        
        Args:
            html_content: HTML content to extract title from
            
        Returns:
            Page title or URL if not found
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            
            # Fallback to h1 if title tag not found
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.string:
                return h1_tag.string.strip()
                
            return "Untitled Document"
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return "Untitled Document"
    
    def _clean_content(self, html_content: str) -> str:
        """
        Clean and extract text content from HTML.
        
        Args:
            html_content: HTML content to clean
            
        Returns:
            Cleaned text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.extract()
                
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning content: {e}")
            return ""
