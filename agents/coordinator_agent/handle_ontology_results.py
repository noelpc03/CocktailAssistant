"""
Handler for ontology query results in the coordinator agent.
"""
import logging

logger = logging.getLogger(__name__)

async def handle_ontology_results(self, content, message):
    """
    Handle results from ontology agent.
    
    Args:
        content: The message content
        message: The full message
        
    Returns:
        Response message or None
    """
    query = content.get("query", "")
    results = content.get("results", [])
    operation_id = content.get("operation_id", "")
    error = content.get("error", None)  # Verificar si hay error en la consulta
    
    if error:
        logger.warning(f"Error en consulta ontológica para '{query}': {error}")
        
    logger.info(f"Handling ontology results for query: {query}")
    
    # Find the operation
    operation = None
    if operation_id in self.active_operations:
        operation = self.active_operations[operation_id]
    else:
        # Try to find by query as fallback
        for op_id, op in self.active_operations.items():
            if op.get("type") == "search" and op.get("query") == query:
                operation = op
                operation_id = op_id
                break
    
    if not operation:
        # No matching operation found
        logger.warning(f"No operation found for ontology results with ID {operation_id}")
        return None
    
    # Si hay error o no hay resultados, activar crawler dinámico
    # Verificar si hay un error específico o si los resultados están vacíos
    if error or not results or (isinstance(results, list) and len(results) == 0):
        logger.info(f"La consulta a la ontología falló o retornó vacía. Activando crawler dinámico para '{query}'")
        
        # Determinar si el error es de sintaxis SPARQL
        is_syntax_error = False
        error_details = ""
        if error:
            error_lower = error.lower() if isinstance(error, str) else ""
            syntax_keywords = ["syntax", "parse", "expected", "found", "sparql", "unterminated", "error"]
            is_syntax_error = any(keyword in error_lower for keyword in syntax_keywords)
            error_details = f"Error en consulta ontológica: {error}"
            
            # Log específico para errores de sintaxis SPARQL
            if is_syntax_error:
                logger.warning(f"Error de sintaxis SPARQL detectado: {error}. Activando crawler dinámico como plan de contingencia.")
                print(f"[Coordinator] Error de sintaxis SPARQL detectado. Activando crawler dinámico como fallback.")
        
        # Forzar uso de crawler dinámico independientemente del tipo de error
        operation["needs_dynamic_crawling"] = True
        operation["ontology_failed"] = True  # Marcar explícitamente que la ontología falló
        
        # Para problemas con cockteles específicos conocidos, agregamos una nota para facilitar debugging
        if "aperol" in query.lower() or "spritz" in query.lower():
            logger.info(f"Detectada consulta sobre Aperol Spritz. Este es un caso conocido con posibles problemas sintácticos.")
            print(f"[Coordinator] Detectada consulta sobre Aperol Spritz. Usando crawler dinámico para obtener información confiable.")
        
        # Enviar una respuesta al generation_agent para que active el crawler dinámico
        if operation.get("use_llm", True):
            await self.send_message("generation_agent", {
                "action": "generate_response",
                "query": query,
                "context": error_details if error else "No se encontraron resultados en la ontología.",
                "operation_id": operation_id,
                "needs_dynamic_crawling": True,  # Explícitamente indicar que necesita crawler dinámico
                "ontology_failed": True,  # Marcar que la ontología falló
                "search_results": []  # Resultados vacíos
            })
            logger.info(f"Enviada solicitud a generation_agent para activar crawler dinámico para '{query}'")
            return None
    
    # Update operation status
    self.active_operations[operation_id]["status"] = "completed"
    self.active_operations[operation_id]["results"] = results
    
    # Check if use_llm is True and results exist
    use_llm = operation.get("use_llm", False)
    
    if use_llm and results:
        # Format results for LLM processing
        context = format_ontology_results_for_llm(results)
        
        # Send to generation agent
        await self.send_message("generation_agent", {
            "action": "generate_response",
            "query": query,
            "context": context,
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

def format_ontology_results_for_llm(results):
    """
    Format ontology results for the LLM.
    
    Args:
        results: List of ontology query results
        
    Returns:
        Formatted context string
    """
    context = []
    
    for result in results:
        result_type = result.get("type", "")
        
        if result_type == "cocktail_with_ingredient":
            context.append(f"Cocktail {result.get('cocktail', '')} contains {result.get('ingredient', '')}.")
        elif result_type == "ingredient_in_cocktail":
            context.append(f"{result.get('cocktail', '')} contains {result.get('ingredient', '')}.")
        elif result_type == "glass_for_cocktail":
            context.append(f"{result.get('cocktail', '')} is served in a {result.get('glass', '')}.")
        elif result_type == "method_for_cocktail":
            context.append(f"{result.get('cocktail', '')} is prepared by {result.get('method', '')}.")
        elif result_type == "cocktail_description":
            context.append(f"{result.get('cocktail', '')}: {result.get('description', '')}")
        elif result_type == "general_match":
            context.append(f"{result.get('subject', '')} {result.get('predicate', '')} {result.get('object', '')}.")
    
    return "\n".join(context)
