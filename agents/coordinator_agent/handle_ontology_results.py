"""
Handler for ontology query results in the coordinator agent.
"""
import logging
import json

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
    try:
        # Log the entire content for debugging
        logger.info(f"Received ontology results with action={content.get('action')}: {json.dumps(content, default=str)[:500]}...")
        print(f"\n{'%'*100}\n[DEBUG-HANDLE] PROCESSING ONTOLOGY RESULTS (action={content.get('action')})\n{'%'*100}", flush=True)
        
        query = content.get("query", "")
        results = content.get("results", [])
        operation_id = content.get("operation_id", "")
        error = content.get("error", None)  # Verificar si hay error en la consulta
        status = content.get("status", "unknown")
        analysis = content.get("analysis", {})
        sparql_query = content.get("sparql_query", "")
        
        # Also check for error in answer
        answer = content.get("answer", "")
        
        print(f"QUERY: '{query}'", flush=True)
        print(f"OPERATION ID: {operation_id}", flush=True)
        print(f"STATUS: {status}", flush=True)
        print(f"RESULTS: {type(results)} con {len(results) if isinstance(results, list) else '?'} elementos", flush=True)
        print(f"ERROR: {error}", flush=True)
        print(f"ANSWER: {answer}", flush=True)
        
        print(f"\nANALYSIS SUMMARY:", flush=True)
        if isinstance(analysis, dict):
            print(f"  Intent: {analysis.get('intent', 'unknown')}", flush=True)
            print(f"  Entities: {json.dumps(analysis.get('entities', []), ensure_ascii=False)}", flush=True)
        else:
            print(f"  Analysis not available or not in expected format: {type(analysis)}", flush=True)
            
        print(f"\nSPARQL QUERY:", flush=True)
        print(f"{sparql_query[:500]}..." if len(sparql_query) > 500 else sparql_query, flush=True)
        
        print(f"\nFIRST 3 RESULTS:", flush=True)
        if results and len(results) > 0:
            print(json.dumps(results[:3], indent=2, ensure_ascii=False, default=str), flush=True)
            if len(results) > 3:
                print(f"...and {len(results) - 3} more results", flush=True)
        else:
            print("NO RESULTS FOUND", flush=True)
            
        print(f"{'%'*100}\n", flush=True)
        
        if error:
            logger.warning(f"Error en consulta ontológica para '{query}': {error}")
        
        if status == "error":
            logger.warning(f"Estado de error en resultados de ontología para '{query}': {content.get('message', 'No message')}")
            
        logger.info(f"Handling ontology results for query: '{query}' with status: {status}")
    except Exception as e:
        logger.error(f"Exception while handling ontology results: {str(e)}")
        # Set defaults to continue processing
        query = content.get("query", "Unknown query")
        results = []
        operation_id = content.get("operation_id", "unknown_operation")
        error = str(e)
        status = "error"
    
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
        print(f"[DEBUG-HANDLE] Activando crawler dinámico para '{query}' debido a resultados vacíos o error", flush=True)
        
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
        
        # Enviar solicitud directamente al agente de crawler dinámico
        try:
            # Solicitar información de la web directamente
            await self.send_message("dynamic_crawler_agent", {
                "action": "search_web",
                "query": query,
                "operation_id": operation_id,
                "source": "ontology_failed"
            })
            
            logger.info(f"Solicitud enviada al crawler dinámico para '{query}'")
            print(f"[DEBUG-HANDLE] Solicitud enviada al crawler dinámico", flush=True)
            
            # También notificar al agente generador sobre el fallo de la ontología
            message_to_generation = {
                "action": "generate_response",
                "query": query,
                "context": error_details if error else "No se encontraron resultados en la ontología.",
                "operation_id": operation_id,
                "needs_dynamic_crawling": True,  # Explícitamente indicar que necesita crawler dinámico
                "ontology_failed": True,  # Marcar que la ontología falló
                "search_results": []  # Resultados vacíos
            }
            
            await self.send_message("generation_agent", message_to_generation)
            print(f"[DEBUG-HANDLE] ✓ Mensaje enviado correctamente a generation_agent", flush=True)
            return None
            
        except Exception as e:
            print(f"[DEBUG-HANDLE] ❌ Error activando crawler dinámico: {str(e)}", flush=True)
            logger.error(f"Error activando crawler dinámico: {str(e)}")
            
            # Fallback - enviar directamente al generation_agent como antes
            message_to_generation = {
                "action": "generate_response",
                "query": query,
                "context": error_details if error else "No se encontraron resultados en la ontología.",
                "operation_id": operation_id,
                "needs_dynamic_crawling": True,
                "ontology_failed": True,
                "search_results": []
            }
            
            try:
                await self.send_message("generation_agent", message_to_generation)
                return None
            except Exception as e2:
                logger.error(f"Error enviando mensaje a generation_agent: {str(e2)}")
                return None
    
    # Update operation status
    self.active_operations[operation_id]["status"] = "completed"
    self.active_operations[operation_id]["results"] = results
    
    # Check if use_llm is True and results exist
    use_llm = operation.get("use_llm", False)
    
    if use_llm and results:
        # Enviar directamente los resultados crudos de la ontología al agente generador
        # Esto elimina la generación intermedia de respuestas y permite que el LLM
        # sea utilizado solo una vez por el agente generador
        
        # Prepara un mensaje mejorado con los resultados crudos
        generation_message = {
            "action": "generate_response",
            "query": query,
            "ontology_results": results,  # Enviamos los resultados crudos de SPARQL
            "operation_id": operation_id
        }
        
        print(f"[DEBUG-HANDLE] Enviando resultados crudos de ontología a generation_agent: {json.dumps(generation_message, default=str)[:500]}", flush=True)
        
        try:
            await self.send_message("generation_agent", generation_message)
            print(f"[DEBUG-HANDLE] ✓ Mensaje enviado correctamente a generation_agent con resultados crudos de ontología", flush=True)
        except Exception as e:
            print(f"[DEBUG-HANDLE] ❌ Error enviando mensaje a generation_agent: {str(e)}", flush=True)
        
        # No response yet, wait for generation
        print(f"[DEBUG-HANDLE] Retornando None después de enviar mensaje a generation_agent", flush=True)
        return None
    else:
        # Send results directly to requester
        response = {
            "recipient": operation["requester"],
            "content": {
                "action": "search_response",
                "status": "success",
                "query": query,
                "results": results
            }
        }
        print(f"[DEBUG-HANDLE] Enviando resultados directamente al solicitante: {json.dumps(response, default=str)[:500]}", flush=True)
        return response

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
