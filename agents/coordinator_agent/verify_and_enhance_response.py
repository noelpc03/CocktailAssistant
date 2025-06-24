"""
Module for checking if web enhancement is needed for responses.
"""
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

async def verify_and_enhance_response(self, query: str, response: str, operation_id: str) -> str:
    """
    Determine if a response needs enhancement based on operation flags,
    and return the appropriate enhanced response when necessary.
    
    Args:
        query: User query
        response: Generated response
        operation_id: Operation ID
        
    Returns:
        Enhanced response if web information is available, or original response
    """
    try:
        start_time = time.time()
        # Check if this operation has web information available
        operation = self.active_operations.get(operation_id, {})
        
        # Check if we have web_info in the operation
        web_info = operation.get("web_info", "")
        
        # If there's no web info, we need to check if the response needs enhancement
        if not web_info:
            logger.info(f"No web information available yet for query: '{query}'. Checking response quality.")
            
            # Check if the operation requires web info (dynamic crawling)
            needs_dynamic_crawling = operation.get("needs_dynamic_crawling", False)
            
            # Check if response is empty or has errors
            is_empty_response = not response or response.strip() == "" or "No se encontraron resultados" in response
            has_error_message = "error" in response.lower() or "falló" in response.lower() or "no está disponible" in response.lower()
            
            # If the response has issues or we know we need web info, get it
            if is_empty_response or has_error_message or needs_dynamic_crawling:
                reason = "Empty response" if is_empty_response else ("Error message detected" if has_error_message else "Dynamic crawling required")
                logger.info(f"Response needs enhancement for query: '{query}'. Reason: {reason}")
                logger.info(f"Searching the web for additional information...")
                
                try:
                    # Search the web with an increased timeout (30 segundos)
                    search_task = asyncio.create_task(asyncio.to_thread(self.dynamic_crawler.search_web, query, 5))
                    search_results = await asyncio.wait_for(search_task, timeout=30.0)
                    
                    logger.info(f"Web search completed in {time.time() - start_time:.2f}s")
                    
                    if not search_results:
                        logger.warning(f"No search results found for query: '{query}'")
                        return response
                        
                    # Extract relevant information from search results with an increased timeout (40 seconds)
                    extract_task = asyncio.create_task(asyncio.to_thread(self.dynamic_crawler.extract_relevant_info, query, search_results))
                    web_info = await asyncio.wait_for(extract_task, timeout=40.0)
                    
                    logger.info(f"Information extraction completed in {time.time() - start_time:.2f}s")
                    
                    if not web_info:
                        logger.warning(f"No relevant information extracted for query: '{query}'")
                        return response
                        
                    # Enhance the response with web information with an increased timeout (20 seconds)
                    enhance_task = asyncio.create_task(asyncio.to_thread(self.dynamic_crawler.enhance_response, query, response, web_info))
                    enhanced_response = await asyncio.wait_for(enhance_task, timeout=20.0)
                    
                    total_time = time.time() - start_time
                    logger.info(f"Response enhanced with dynamic web information for query: '{query}' (total time: {total_time:.2f}s)")
                    
                    return enhanced_response
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout during web crawling operations after {time.time() - start_time:.2f}s. Returning original response.")
                    return response
                    
                except Exception as e:
                    logger.error(f"Error during web enhancement: {e}")
                    return response
            
            # If the response seems good but we don't have web info, just return the original
            logger.info(f"Response seems adequate for query: '{query}'. No enhancement needed.")
            return response
        
        # If we already have web_info in the operation
        # Check if response is empty or has errors
        is_empty_response = not response or response.strip() == "" or "No se encontraron resultados" in response
        has_error_message = "error" in response.lower() or "falló" in response.lower() or "no está disponible" in response.lower()
        
        # If the response is problematic and we have web info, use it to generate a completely new response
        if is_empty_response or has_error_message:
            logger.info(f"Respuesta problemática detectada. Usando información web para: '{query}'")
            return await asyncio.to_thread(self.dynamic_crawler.generate_response_from_scratch, query, web_info)
        
        # If the response seems good but we have web info, just return the original response
        # since the web info was already considered in the generation process
        logger.info(f"Respuesta adecuada para query: '{query}'")
        return response
        
    except Exception as e:
        logger.error(f"Error in verify_and_enhance_response: {e}")
        return response  # Return original response on error
