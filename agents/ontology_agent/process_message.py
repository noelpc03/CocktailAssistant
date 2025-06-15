"""
Implementation of process_message for the OntologyAgent class.
"""
import logging
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
    content = message.get("content", {})
    action = content.get("action")
    
    if action == "query_ontology":
        # Process an ontology query
        query = content.get("query", "")
        operation_id = content.get("operation_id", "")
        
        if not query:
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "ontology_results",
                    "status": "error",
                    "message": "Empty query",
                    "operation_id": operation_id
                }
            }
        
        # Process the query against the ontology
        try:
            results = self.query_ontology(query)
            
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "ontology_results",
                    "status": "success",
                    "query": query,
                    "results": results,
                    "operation_id": operation_id
                }
            }
        except Exception as e:
            logger.error(f"Error querying ontology: {e}")
            return {
                "recipient": message.get("sender", "coordinator_agent"),
                "content": {
                    "action": "ontology_results",
                    "status": "error",
                    "message": f"Error querying ontology: {str(e)}",
                    "operation_id": operation_id
                }
            }
    else:
        logger.warning(f"Unknown action '{action}' received by ontology agent")
        return {
            "recipient": message.get("sender", "coordinator_agent"),
            "content": {
                "action": "error",
                "error": f"Unknown action: {action}",
                "operation_id": content.get("operation_id", "unknown")
            }
        }
