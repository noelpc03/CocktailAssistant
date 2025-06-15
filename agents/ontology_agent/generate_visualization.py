"""
Method for generating ontology visualizations.
"""
import os
import logging
from rdflib import Graph
from .visualization import generate_graph_visualization, extract_ontology_subset

logger = logging.getLogger(__name__)

def generate_ontology_visualization(graph: Graph, output_dir: str, filename: str = "ontology_visualization", format: str = "png") -> str:
    """
    Generate a visualization of the ontology.
    
    Args:
        graph: The RDF graph to visualize
        output_dir: Directory to save visualization
        filename: Base name for the output file (without extension)
        format: Image format (png, svg, pdf)
        
    Returns:
        Path to generated visualization file or None if failed
    """
    try:
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate full path for output file
        output_path = os.path.join(output_dir, filename)
        
        # Extract a subset of the graph to avoid overwhelming visualization
        subset_graph = extract_ontology_subset(graph, limit=200)
        
        # Generate visualization
        output_file = generate_graph_visualization(subset_graph, output_path, format)
        
        # Return path to visualization file
        return output_file
    except Exception as e:
        logger.error(f"Error generating ontology visualization: {e}")
        return None
