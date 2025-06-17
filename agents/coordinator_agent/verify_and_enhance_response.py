"""
Module for verification and enhancement of responses using dynamic crawling.
"""
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

async def verify_and_enhance_response(self, query: str, response: str, operation_id: str) -> str:
    """
    Verify if a response is adequate, and if not, enhance it using dynamic web crawling.
    This function has un timeout más amplio para dar suficiente tiempo al crawler
    para completar su trabajo cuando la ontología falla.
    
    Args:
        query: User query
        response: Generated response
        operation_id: Operation ID
        
    Returns:
        Enhanced response if needed, or original response if adequate
    """
    try:
        # Registrar tiempo de inicio para monitorear tiempos de ejecución
        start_time = time.time()
        
        # Check if dynamic crawling is needed for this operation
        operation = self.active_operations.get(operation_id, {})
        needs_dynamic_crawling = operation.get("needs_dynamic_crawling", False)
        
        # Si la respuesta es vacía, indica que no hay resultados, o contiene mensajes de error, activamos el crawler dinámico
        is_empty_response = not response or response.strip() == "" or "No se encontraron resultados" in response
        has_error_message = "error" in response.lower() or "falló" in response.lower() or "no está disponible" in response.lower()
        
        if is_empty_response or has_error_message:
            logger.info(f"Respuesta problemática detectada para: '{query}'. Activando crawler dinámico.")
            # Forzar el uso del crawler dinámico para consultas con problemas
            needs_dynamic_crawling = True
        elif not needs_dynamic_crawling:
            # Skip verification if dynamic crawling was not requested and response is not problematic
            logger.info(f"Skipping dynamic crawling for query: '{query}' (not needed)")
            return response
            
        logger.info(f"Verificando respuesta para query: '{query}'")
        
        is_adequate = False
        reason = ""
        if not is_empty_response:
            # First, verify if the response is adequate (solo si no es una respuesta vacía)
            try:
                # Usar un timeout más amplio para la verificación (10 segundos)
                verify_task = asyncio.create_task(asyncio.to_thread(self.dynamic_crawler.verify_response, query, response))
                is_adequate, reason = await asyncio.wait_for(verify_task, timeout=10.0)
                
                if is_adequate:
                    logger.info(f"Response verified as ADEQUATE for query: '{query}'")
                    return response
            except asyncio.TimeoutError:
                logger.warning(f"Timeout during response verification. Proceeding with web search.")
                is_adequate = False
                reason = "Timeout during verification"
        else:
            # Si la respuesta está vacía, consideramos que no es adecuada
            logger.info(f"Respuesta vacía detectada, considerada como INADEQUATE")
            is_adequate = False
            reason = "No se encontraron resultados en la base de conocimiento"
            
        # If inadequate, search the web for additional information
        logger.info(f"Response verified as INADEQUATE for query: '{query}'. Reason: {reason}")
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
        logger.error(f"Error in verify_and_enhance_response: {e}")
        return response  # Return original response on error
