"""
Implementation of process_message for the OntologyAgent class.
"""
import logging
import os
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an incoming message and return a response.
    
    Args:
        message: The message to process
        
    Returns:
        The response message
    """
    print(f"[OntologyAgent] Recibido mensaje: {message}")
    logger.info(f"OntologyAgent received message: {message}")
    
    content = message.get("content", {})
    action = content.get("action")
    print(f"[OntologyAgent] Procesando acción: {action}")
    
    if action == "extract_ontology":
        operation_id = content.get("operation_id", "unknown_op_id")
        search_results = content.get("search_results")
        
        logger.info(f"Received extract_ontology request with operation_id: {operation_id}")
        print(f"[Ontology] Iniciando proceso de extracción de ontología (operation_id: {operation_id})")
        print(f"[Ontology] Remitente del mensaje: {message.get('sender', 'unknown')}")
        
        # Intentar cargar tripletas existentes primero
        existing_triples = self._load_extracted_triples()
        if existing_triples:
            print(f"[Ontology] Cargadas {len(existing_triples)} tripletas previamente extraídas")
            logger.info(f"Loaded {len(existing_triples)} previously extracted triples")
        
        if not search_results:
            # Try to get documents from data store
            search_results = self.data_store.get("search_results", [])
            print(f"[Ontology] Buscando documentos en data_store: {len(search_results)} encontrados")
        
        if not search_results:
            # Cargar desde metadata.json
            metadata = self.data_store.load_metadata()
            if metadata:
                print(f"[Ontology] Cargando desde metadata.json: {len(metadata)} documentos encontrados")
                search_results = metadata
                
        if not search_results:
            print(f"[Ontology] ❌ Error: No hay documentos disponibles para extracción de ontología")
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "ontology_results",
                    "operation_id": operation_id,
                    "status": "error",
                    "message": "No documents available for ontology extraction"
                }
            }
        
        print(f"[Ontology] Procesando {len(search_results)} documentos en total")
        
        print(f"[Ontology] Procesando TODOS los {len(search_results)} documentos disponibles")
        
        # Extract and process triples
        try:
            # Si ya tenemos tripletas previas, usarlas como base
            if existing_triples:
                print(f"[Ontology] Continuando extracción con {len(existing_triples)} tripletas existentes")
                extracted_triples = existing_triples
                
                # Procesar solo documentos nuevos
                extracted_triples.extend(await self.extract_triples_from_documents(search_results))
            else:
                # Comenzar desde cero
                extracted_triples = await self.extract_triples_from_documents(search_results)
            
            # Apply reasoning to infer new triples
            print(f"[Ontology] Aplicando razonamiento a {len(extracted_triples)} tripletas")
            self.apply_reasoning()
            
            # Generate ontology schema
            print(f"[Ontology] Generando esquema de la ontología")
            schema = self.generate_ontology_schema()
            print(f"[Ontology] Esquema generado con {len(schema['classes'])} clases, {len(schema['objectProperties'])} propiedades de objeto, y {len(schema['dataProperties'])} propiedades de datos")
            
            # Save ontology
            ontology_file = self.save_ontology()
            print(f"[Ontology] Ontología guardada en {ontology_file}")
            
            # Asegurarse de que el mensaje tiene toda la información necesaria
            response = {
                "sender": "ontology_agent",
                "recipient": message["sender"],
                "content": {
                    "action": "ontology_results",
                    "operation_id": operation_id,
                    "status": "success",
                    "message": f"Ontology extraction completed with {len(extracted_triples)} triples",
                    "ontology_file": ontology_file,
                    "triple_count": len(extracted_triples)
                }
            }
            
            print(f"[OntologyAgent] Mensaje de respuesta completo: {response}")
            return response
        except Exception as e:
            error_message = f"Error durante la extracción de ontología: {str(e)}"
            print(f"[Ontology] ❌ {error_message}")
            logger.error(error_message)
            return {
                "sender": "ontology_agent",
                "recipient": message["sender"],
                "content": {
                    "action": "ontology_results",
                    "operation_id": operation_id,
                    "status": "error",
                    "message": error_message
                }
            }
    elif action == "visualize_ontology":
        operation_id = content.get("operation_id")
        
        # Generate visualization
        visualization_file = self.generate_visualization()
        
        if visualization_file:
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "visualization_results",
                    "operation_id": operation_id,
                    "status": "success",
                    "visualization_file": visualization_file
                }
            }
        else:
            return {
                "recipient": message["sender"],
                "content": {
                    "action": "visualization_results",
                    "operation_id": operation_id,
                    "status": "error",
                    "message": "Failed to generate visualization"
                }
            }
    elif action == "query_ontology":
        operation_id = content.get("operation_id", "unknown_op_id")
        query_text = content.get("query", "")
        language = content.get("language", None)
        
        if not query_text:
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "query_results",
                    "operation_id": operation_id,
                    "status": "error",
                    "message": "No query text provided"
                }
            }
        
        try:
            print(f"[Ontology] Procesando consulta: {query_text}")
            logger.info(f"Processing ontology query: {query_text}")
            
            # Process the query
            result = await self.process_query(query_text, language)
            
            # Return the result
            response = {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "query_results",
                    "operation_id": operation_id,
                    "status": "success",
                    "query": query_text,  # Asegurar que la consulta original está incluida
                    **result  # Include all result fields in the response
                }
            }
            
            # Imprimir la respuesta completa enviada al coordinador
            print(f"\n{'@'*100}\n[DEBUG-AGENT] COMPLETE RESPONSE FROM PROCESS_MESSAGE\n{'@'*100}", flush=True)
            print(f"QUERY: {query_text}", flush=True)
            print(f"OPERATION ID: {operation_id}", flush=True)
            print(f"RESPONSE STATUS: {response['content'].get('status', 'N/A')}", flush=True)
            print(f"ANSWER: {result.get('answer', '[No answer provided]')}", flush=True)
            print(f"\nFULL RESPONSE:", flush=True)
            print(json.dumps(response, indent=2, default=str, ensure_ascii=False)[:2000], flush=True)
            print(f"{'@'*100}\n", flush=True)
            
            print(f"[Ontology] Consulta procesada con éxito")
            return response
            
        except Exception as e:
            error_message = f"Error al procesar la consulta de ontología: {str(e)}"
            print(f"[Ontology] ❌ {error_message}")
            logger.error(error_message)
            
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "query_results",
                    "operation_id": operation_id,
                    "status": "error",
                    "message": error_message,
                    "answer": "Lo siento, ocurrió un error al procesar tu consulta."
                }
            }
    else:
        # Para cualquier otra acción no implementada
        logger.warning(f"Action '{action}' not yet implemented in ontology agent")
        return {
            "recipient": message.get("sender", "coordinator_agent"),
            "content": {
                "action": "error",
                "error": f"Action not implemented: {action}",
                "operation_id": content.get("operation_id", "unknown")
            }
        }
