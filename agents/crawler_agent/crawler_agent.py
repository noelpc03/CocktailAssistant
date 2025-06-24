"""
Crawler agent responsible for fetching web content using Ant Colony Optimization algorithm.
"""
import asyncio
import logging
import re
import requests
import time
import random
import urllib.parse
from typing import Dict, Any, List, Set, Tuple, DefaultDict
from collections import deque, defaultdict
from bs4 import BeautifulSoup

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore

logger = logging.getLogger(__name__)

class AntColonyCrawler:
    """Ant Colony Optimization algorithm for web crawling"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ACO crawler"""
        self.config = config.get("ant_colony", {})
        self.num_ants = self.config.get("num_ants", 8)
        self.alpha = self.config.get("alpha", 1.0)  # Pheromone importance
        self.beta = self.config.get("beta", 2.0)  # Heuristic importance
        self.evap_rate = self.config.get("evaporation_rate", 0.1)
        self.deposit_factor = self.config.get("deposit_factor", 1.0)
        
        # Keywords and trusted domains
        self.cocktail_keywords = self.config.get("cocktail_keywords", [])
        self.trusted_domains = self.config.get("trusted_domains_weight", {})
        self.blacklist_patterns = self.config.get("blacklist_patterns", [])
        
        # Pheromone matrix: URL -> pheromone level
        self.pheromones = defaultdict(lambda: 0.1)  # Default pheromone level of 0.1
        
        # Track URL quality for updating pheromones
        self.url_quality = {}
        
    def calculate_heuristic(self, url: str, anchor_text: str = "") -> float:
        """
        Calculate heuristic value for a URL based on keywords and domain.
        
        Args:
            url: URL to evaluate
            anchor_text: Text of the link (optional)
            
        Returns:
            Heuristic value between 0.0 and 1.0
        """
        url_lower = url.lower()
        score = 0.1  # Base score
        
        # Check for blacklisted patterns
        for pattern in self.blacklist_patterns:
            if pattern.lower() in url_lower:
                return 0.01  # Very low score for blacklisted URLs
        
        # Score for cocktail keywords in URL
        for keyword in self.cocktail_keywords:
            if keyword.lower() in url_lower:
                score += 0.2
        
        # Score for cocktail keywords in anchor text
        if anchor_text:
            anchor_lower = anchor_text.lower()
            for keyword in self.cocktail_keywords:
                if keyword.lower() in anchor_lower:
                    score += 0.3
        
        # Score for trusted domains
        domain = urllib.parse.urlparse(url).netloc
        for trusted_domain, weight in self.trusted_domains.items():
            if trusted_domain in domain:
                score *= weight
                break
        
        return min(1.0, score)  # Cap at 1.0
    
    def select_next_urls(self, candidates: List[Tuple[str, str]], visited: Set[str]) -> List[Tuple[str, str]]:
        """
        Select next URLs to visit using ACO algorithm.
        
        Args:
            candidates: List of (url, anchor_text) tuples
            visited: Set of already visited URLs
            
        Returns:
            List of selected (url, anchor_text) tuples
        """
        if not candidates:
            return []
        
        # Filter out already visited URLs and apply domain diversity
        candidates = [(url, anchor) for url, anchor in candidates if url not in visited]
        if not candidates:
            return []
            
        # Group candidates by domain to ensure diversity
        domains = defaultdict(list)
        for url, anchor in candidates:
            domain = urllib.parse.urlparse(url).netloc
            domains[domain].append((url, anchor))
            
        # Calculate probabilities for each URL
        url_probabilities = []
        for url, anchor_text in candidates:
            pheromone = self.pheromones[url]
            heuristic = self.calculate_heuristic(url, anchor_text)
            
            # ACO formula: probability ∝ (pheromone)^α * (heuristic)^β
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
            url_probabilities.append((url, anchor_text, probability))
        
        # Sort by probability (highest first)
        url_probabilities.sort(key=lambda x: x[2], reverse=True)
        
        # Implement diverse selection strategy
        selected = []
        domain_count = defaultdict(int)
        max_per_domain = max(1, self.num_ants // 3)  # Limit URLs per domain
        
        # First strategy: Take some top URLs based on highest probability
        top_n = max(1, min(len(url_probabilities) // 2, self.num_ants // 2))
        for url, anchor_text, _ in url_probabilities[:top_n]:
            domain = urllib.parse.urlparse(url).netloc
            if domain_count[domain] < max_per_domain:
                selected.append((url, anchor_text))
                domain_count[domain] += 1
                if len(selected) >= self.num_ants:
                    break
        
        # Second strategy: Use roulette wheel selection for remaining slots
        remaining_candidates = [item for item in url_probabilities if (item[0], item[1]) not in selected]
        remaining_slots = self.num_ants - len(selected)
        
        for _ in range(min(remaining_slots, len(remaining_candidates))):
            if not remaining_candidates:
                break
                
            # Calculate total probability
            total_prob = sum(prob for _, _, prob in remaining_candidates)
            if total_prob <= 0:
                break
                
            # Normalize probabilities
            normalized = [(url, anchor, prob/total_prob) for url, anchor, prob in remaining_candidates]
            
            # Roulette wheel selection
            r = random.random()
            cumulative = 0
            selected_idx = -1
            
            for i, (url, anchor, prob) in enumerate(normalized):
                # Check domain diversity
                domain = urllib.parse.urlparse(url).netloc
                if domain_count[domain] < max_per_domain:
                    cumulative += prob
                    if r <= cumulative:
                        selected_idx = i
                        break
            
            # If we didn't select anything due to domain constraints, try again without constraints
            if selected_idx < 0:
                for i, (_, _, prob) in enumerate(normalized):
                    cumulative += prob
                    if r <= cumulative:
                        selected_idx = i
                        break
                        
            # If still no selection, pick the highest probability
            if selected_idx < 0 and normalized:
                selected_idx = 0
            elif selected_idx < 0:
                break
                
            # Add to selected and remove from candidates
            url, anchor_text, _ = remaining_candidates[selected_idx]
            domain = urllib.parse.urlparse(url).netloc
            selected.append((url, anchor_text))
            domain_count[domain] += 1
            
            # Remove from candidates
            remaining_candidates.pop(selected_idx)
        
        logger.debug(f"ACO selected {len(selected)} URLs out of {len(candidates)} candidates")
        return selected
    
    def update_pheromones(self, url: str, quality: float) -> None:
        """
        Update pheromone levels for a URL based on content quality.
        
        Args:
            url: The URL to update
            quality: Quality score between 0.0 and 1.0
        """
        self.url_quality[url] = quality
        deposit = quality * self.deposit_factor
        self.pheromones[url] += deposit
    
    def evaporate_pheromones(self) -> None:
        """Apply pheromone evaporation to all URLs"""
        # Log highest pheromone levels before evaporation for debugging
        if len(self.pheromones) > 0:
            top_urls = sorted(self.pheromones.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.debug(f"Top 5 pheromone levels before evaporation: {top_urls}")
            
        # Apply evaporation
        for url in self.pheromones:
            self.pheromones[url] *= (1 - self.evap_rate)


class CrawlerAgent(Agent):
    """Agent responsible for crawling web content using Ant Colony Optimization"""
    
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
        
        # Initialize rate limiting parameters
        self.request_timestamps = {}
        self.request_delay = self.config.get("request_delay", 1.0)  # Delay between requests to same domain in seconds
        
        # Initialize Ant Colony Optimization
        self.ant_colony = AntColonyCrawler(self.config)
        
        # Initialize URL relevance parameters from ant colony config
        self.relevance_keywords = self.ant_colony.cocktail_keywords
        
        # Wikipedia specific patterns
        self.wiki_patterns = {
            'article': re.compile(r'/wiki/(?!Special:|Talk:|User:|Template:|Category_talk:|Help:|Portal:|File:)'),
            'language': re.compile(r'//([a-z]{2,3})\.wikipedia\.org'),
            'disambiguation': re.compile(r'disambiguation|desambiguación', re.IGNORECASE)
        }
    
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
            # Si no se proporcionaron URLs específicas, usar las URLs predeterminadas de la configuración
            urls = content.get("urls")
            if not urls or len(urls) == 0:
                urls = self.config.get("default_urls", [])
            
            operation_id = content.get("operation_id")
            
            logger.info(f"Received crawl_urls request with operation_id: {operation_id}")
            logger.info(f"URLs to crawl: {urls}")
            logger.info(f"Config default_urls: {self.config.get('default_urls', [])}")
            
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
    
    async def process_url(self, url: str, depth: int, anchor_text: str, visited_urls: set, min_relevance: float) -> Dict[str, Any]:
        """
        Process a single URL: fetch, analyze, and extract links.
        
        Args:
            url: URL to process
            depth: Current depth level
            anchor_text: Anchor text of the link
            visited_urls: Set of already visited URLs
            min_relevance: Minimum relevance threshold
            
        Returns:
            Dictionary with processing results
        """
        # Default relevance until content is analyzed
        relevance = self.ant_colony.calculate_heuristic(url, anchor_text)
        result = {
            "url": url,
            "depth": depth,
            "success": False,
            "document": None,
            "links": [],
            "relevance": relevance
        }
            
        try:
            logger.info(f"Crawling URL: {url} (depth: {depth}, relevance: {relevance:.2f})")
            
            # Apply rate limiting
            await self._respect_rate_limits(url)
            
            content = await self._fetch_page(url)
            
            if not content:
                logger.warning(f"Failed to extract content from {url}")
                return result
                
            # Process current page content
            title = self._extract_title(content)
            clean_content = self._clean_content(content)
            
            # Calculate content relevance
            content_relevance = self._calculate_content_relevance(title, clean_content)
            
            # Update the relevance variable with the actual content relevance
            relevance = content_relevance
            
            doc = {
                "url": url,
                "title": title,
                "content": clean_content,
                "depth": depth,
                "relevance": content_relevance,
                "source": "web_crawl",
                "timestamp": time.time()
            }
            
            # Check if document has sufficient relevance
            if content_relevance >= min_relevance:
                result["document"] = doc
                logger.info(f"Successfully crawled {url} (relevance: {content_relevance:.2f})")
            
            # Extract links with anchor text for better relevance assessment
            if depth < self.config.get("max_depth", 2):
                result["links"] = self._extract_links_with_anchor(content, url)
            
            result["success"] = True
            result["relevance"] = content_relevance
            
            return result
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return result

    async def crawl_urls(self, seed_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Crawl websites using Ant Colony Optimization algorithm starting from seed URLs,
        with concurrent URL processing.
        
        Args:
            seed_urls: Initial URLs to start crawling from
            
        Returns:
            List of documents with extracted content
        """
        documents = []
        visited_urls = set()
        
        # BFS queue with (url, depth, anchor_text) tuples
        queue = deque([(url, 0, "") for url in seed_urls])  # Seed URLs have empty anchor text
        
        # Set maximum depth for BFS traversal
        max_depth = self.config.get("max_depth", 2)
        
        # Get maximum number of documents to crawl
        max_documents = self.config.get("max_documents", 2000)
        
        # Default minimum relevance threshold 
        min_relevance = 0.2  # Can be adjusted based on requirements
        
        # Set concurrency level (number of URLs to process in parallel)
        concurrency = min(16, self.ant_colony.num_ants * 2)  # Use at most 16 concurrent tasks
        
        logger.info(f"Starting Parallel Ant Colony Optimization crawl with {len(seed_urls)} seed URLs and max depth {max_depth}. " 
                   f"Using {self.ant_colony.num_ants} ants for selection and {concurrency} concurrent workers. "
                   f"Document limit: {max_documents}")
        
        page_count = 0
        
        while queue and len(documents) < max_documents:
            # Phase 1: Select a batch of URLs to process concurrently
            batch = []
            batch_by_domain = defaultdict(int)
            max_per_domain = max(2, concurrency // 4)  # Limit URLs per domain in each batch
            
            # Check if we're close to the document limit and adjust batch size if necessary
            remaining_docs = max_documents - len(documents)
            batch_size = min(concurrency, remaining_docs)
            
            while len(batch) < batch_size and queue:
                # Get next URL from queue
                if not queue:
                    break
                    
                url, depth, anchor = queue.popleft()
                
                # Skip if already visited or beyond max depth
                if url in visited_urls or depth > max_depth:
                    continue
                
                # Apply domain diversity in batch selection
                domain = urllib.parse.urlparse(url).netloc
                if batch_by_domain[domain] >= max_per_domain:
                    # Put back in queue for later processing
                    queue.append((url, depth, anchor))
                    continue
                    
                batch.append((url, depth, anchor))
                batch_by_domain[domain] += 1
                visited_urls.add(url)
                page_count += 1
            
            if not batch:
                continue
            
            # Log batch information
            logger.info(f"Processing batch of {len(batch)} URLs, {sum(batch_by_domain.values())} domains")
                
            # Phase 2: Process URLs in parallel
            tasks = []
            for url, depth, anchor in batch:
                task = self.process_url(url, depth, anchor, visited_urls, min_relevance)
                tasks.append(task)
                
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=False)
            
            # Phase 3: Process results, update pheromones and extract new links (synchronized)
            new_links_count = 0
            
            # First update pheromones for all successfully processed URLs
            for result in results:
                if result["success"]:
                    # Update pheromones based on content relevance
                    self.ant_colony.update_pheromones(result["url"], result["relevance"])
                    
                    # Add document to collection if relevant and we haven't reached the limit
                    if result["document"] and len(documents) < max_documents:
                        documents.append(result["document"])
                    
                    # Check if we've reached the document limit
                    if len(documents) >= max_documents:
                        logger.info(f"Document limit of {max_documents} reached. Stopping crawl process.")
            
            # Apply pheromone evaporation once per batch
            self.ant_colony.evaporate_pheromones()
            
            # Then process extracted links for all URLs
            for result in results:
                if not result["success"] or not result["links"]:
                    continue
                    
                # Use Ant Colony Optimization to select next URLs
                selected_links = self.ant_colony.select_next_urls(result["links"], visited_urls)
                
                # Process selected links
                domain_stats = defaultdict(int)
                filtered_links = []
                
                for link_url, link_anchor in selected_links:
                    if link_url not in visited_urls:
                        # Track domain diversity
                        domain = urllib.parse.urlparse(link_url).netloc
                        domain_stats[domain] += 1
                        
                        # Add to queue with depth and anchor text
                        queue.append((link_url, result["depth"] + 1, link_anchor))
                        filtered_links.append(link_url)
                
                # Log domain diversity for this result
                if filtered_links:
                    logger.info(f"Domain diversity at depth {result['depth']}: {dict(domain_stats)}")
                    logger.info(f"Added {len(filtered_links)} new links from {result['url']} (at depth {result['depth']})")
                    new_links_count += len(filtered_links)
            
            logger.info(f"Batch completed: {len(documents)} total documents, {new_links_count} new links queued")
        
        if len(documents) >= max_documents:
            logger.info(f"Crawl reached document limit of {max_documents}. Crawling process stopped.")
        
        logger.info(f"Parallel crawl completed. Visited {page_count} URLs, extracted {len(documents)} documents")
        
        # Count documents by depth level to provide better insights
        depth_counts = {}
        for doc in documents:
            depth_level = doc.get("depth", 0)
            if depth_level in depth_counts:
                depth_counts[depth_level] += 1
            else:
                depth_counts[depth_level] = 1
                
        for depth_level in sorted(depth_counts.keys()):
            logger.info(f"Depth {depth_level}: {depth_counts[depth_level]} documents")
        
        # Store the crawled documents
        self.data_store.set("crawled_documents", documents)
        
        return documents
    
    async def _respect_rate_limits(self, url: str) -> None:
        """
        Respect rate limits for the specific domain.
        
        Args:
            url: URL to check for rate limiting
        """
        domain = urllib.parse.urlparse(url).netloc
        
        # Check if we have a recent request to this domain
        if domain in self.request_timestamps:
            last_request_time = self.request_timestamps[domain]
            elapsed_time = time.time() - last_request_time
            
            # If we need to wait more
            if elapsed_time < self.request_delay:
                wait_time = self.request_delay - elapsed_time
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for domain {domain}")
                await asyncio.sleep(wait_time)
        
        # Update the timestamp after potential waiting
        self.request_timestamps[domain] = time.time()

    async def _fetch_page(self, url: str, max_retries: int = 3) -> str:
        """
        Fetch page content from URL with retries.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retries on failure
            
        Returns:
            HTML content of the page
        """
        retries = 0
        backoff_factor = 1.5  # Exponential backoff factor
        
        while retries < max_retries:
            try:
                # Use a proper user agent to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml',
                    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
                }
                
                # Run the HTTP request in a thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: requests.get(url, headers=headers, timeout=120)  
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:  # Too Many Requests
                    retry_after = min(60, int(response.headers.get('Retry-After', 5)))  # Limitar a máximo 60 segundos
                    logger.warning(f"Rate limited for {url}: waiting {retry_after}s (original: {response.headers.get('Retry-After', 5)}s)")
                    await asyncio.sleep(retry_after)
                    retries += 1
                    continue
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                    return ""
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {url}, retry {retries+1}/{max_retries}")
                retries += 1
                await asyncio.sleep(backoff_factor ** retries)  # Exponential backoff
                continue
                
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return ""
                
        logger.error(f"Failed to fetch {url} after {max_retries} retries")
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
            
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract links from HTML content.
        
        Args:
            html_content: HTML content to extract links from
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs found in the page
        """
        links = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all anchor tags with href attributes
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # Skip empty links, anchors, and javascript links
                if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                
                # Convert relative URLs to absolute
                absolute_url = urllib.parse.urljoin(base_url, href)
                
                # Only include HTTP/HTTPS URLs
                if absolute_url.startswith(('http://', 'https://')):
                    # Normalize URL: remove fragments
                    url_parts = urllib.parse.urlparse(absolute_url)
                    normalized_url = urllib.parse.urlunparse((
                        url_parts.scheme,
                        url_parts.netloc,
                        url_parts.path,
                        url_parts.params,
                        url_parts.query,
                        ''  # Remove fragment
                    ))
                    
                    links.append(normalized_url)
                    
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
            
    def _extract_links_with_anchor(self, html_content: str, base_url: str) -> List[Tuple[str, str]]:
        """
        Extract links with their anchor text from HTML content.
        
        Args:
            html_content: HTML content to extract links from
            base_url: Base URL for resolving relative links
            
        Returns:
            List of tuples (url, anchor_text) found in the page
        """
        links_with_anchor = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all anchor tags with href attributes
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                anchor_text = a_tag.get_text().strip()
                
                # Skip empty links, anchors, and javascript links
                if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                
                # Convert relative URLs to absolute
                absolute_url = urllib.parse.urljoin(base_url, href)
                
                # Only include HTTP/HTTPS URLs
                if absolute_url.startswith(('http://', 'https://')):
                    # Normalize URL: remove fragments
                    url_parts = urllib.parse.urlparse(absolute_url)
                    normalized_url = urllib.parse.urlunparse((
                        url_parts.scheme,
                        url_parts.netloc,
                        url_parts.path,
                        url_parts.params,
                        url_parts.query,
                        ''  # Remove fragment
                    ))
                    
                    links_with_anchor.append((normalized_url, anchor_text))
                    
            return links_with_anchor
            
        except Exception as e:
            logger.error(f"Error extracting links with anchor: {e}")
            return []
        
    def _evaluate_url_relevance(self, url: str, parent_title: str, parent_url: str) -> float:
        """
        Evaluate the relevance of a URL based on its path and text.
        
        Args:
            url: URL to evaluate
            parent_title: Title of the parent page
            parent_url: URL of the parent page
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Parse the URL
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        
        # Base score
        score = 0.0
        
        # Increase score for Wikipedia articles
        if 'wikipedia.org' in domain:
            # Check if it's an article page
            if self.wiki_patterns['article'].search(path):
                score += 0.4
                
                # Extract the article name
                article_name = path.split('/')[-1].replace('_', ' ')
                article_name = urllib.parse.unquote(article_name)
                
                # Check for cocktail-related terms in the article name
                keyword_matches = sum(1 for keyword in self.relevance_keywords 
                                     if keyword.lower() in article_name.lower())
                if keyword_matches > 0:
                    score += min(0.4, keyword_matches * 0.2)  # Up to 0.4 for keyword matches
                    
                # Disambiguation pages are less relevant
                if self.wiki_patterns['disambiguation'].search(article_name):
                    score -= 0.2
                    
                # Prefer same language as parent
                parent_lang_match = self.wiki_patterns['language'].search(parent_url)
                current_lang_match = self.wiki_patterns['language'].search(url)
                
                if parent_lang_match and current_lang_match:
                    if parent_lang_match.group(1) == current_lang_match.group(1):
                        score += 0.1
            else:
                # Non-article Wikipedia pages are less relevant
                score += 0.1
        else:
            # For non-Wikipedia URLs
            # Check if domain contains any relevant keywords
            keyword_matches = sum(1 for keyword in self.relevance_keywords 
                                 if keyword.lower() in domain.lower())
            if keyword_matches > 0:
                score += min(0.3, keyword_matches * 0.1)
                
            # Check if path contains any relevant keywords
            keyword_matches = sum(1 for keyword in self.relevance_keywords 
                                 if keyword.lower() in path.lower())
            if keyword_matches > 0:
                score += min(0.3, keyword_matches * 0.1)
        
        # Ensure the score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))
        
    def _calculate_content_relevance(self, title: str, content: str) -> float:
        """
        Calculate the relevance of content based on keyword matches and text analysis.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Base score
        score = 0.1  # Start with a small base score
        
        # Check title for relevance
        title_lower = title.lower()
        title_keywords = sum(1 for keyword in self.relevance_keywords 
                           if keyword.lower() in title_lower)
        
        # Title keywords are highly relevant
        if title_keywords > 0:
            score += min(0.5, title_keywords * 0.25)  # Up to 0.5 for title keywords
        
        # Penalty for very short content
        if len(content) < 500:
            score *= 0.5
        
        # Use a sample of the content for efficiency
        content_sample = content[:10000]  # First 10000 characters
        content_lower = content_sample.lower()
        
        # Count keyword occurrences
        keyword_counts = {}
        for keyword in self.relevance_keywords:
            # Count standalone occurrences (with word boundaries)
            keyword_lower = keyword.lower()
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            matches = re.findall(pattern, content_lower)
            keyword_counts[keyword] = len(matches)
        
        total_keywords = sum(keyword_counts.values())
        unique_keywords = sum(1 for count in keyword_counts.values() if count > 0)
        
        # Scoring based on keyword density and diversity
        if total_keywords > 0:
            # More unique keywords = higher relevance
            score += min(0.4, unique_keywords * 0.04)
            
            # Higher density = higher relevance, but with diminishing returns
            content_words = len(content_sample.split())
            keyword_density = total_keywords / max(1, content_words)
            score += min(0.2, keyword_density * 200)
            
            # Bonus for high-value keywords (alcohols and specific cocktail terms)
            premium_keywords = ['cocktail', 'cóctel', 'recipe', 'receta', 'gin', 'vodka', 
                              'rum', 'tequila', 'whiskey', 'bourbon', 'martini']
            premium_count = sum(keyword_counts.get(k, 0) for k in premium_keywords)
            if premium_count > 0:
                score += min(0.2, premium_count * 0.05)
        
        # Detect recipe patterns
        recipe_patterns = ['ingredients:', 'ingredientes:', 'preparation:', 'preparación:', 
                          'instructions:', 'instrucciones:', 'directions:', 'steps:']
        for pattern in recipe_patterns:
            if pattern in content_lower:
                score += 0.15
                break
        
        # Ensure the score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))
