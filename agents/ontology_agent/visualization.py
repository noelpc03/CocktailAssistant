"""
Visualization utilities for ontologies.
"""
import os
import logging
import rdflib
from rdflib import Graph
import tempfile
import subprocess

logger = logging.getLogger(__name__)

def generate_graph_visualization(graph: Graph, output_file: str, format: str = "png") -> str:
    """
    Generate visualization of an RDF graph using Graphviz.
    
    Args:
        graph: RDF graph to visualize
        output_file: Base name for output file (without extension)
        format: Output format (png, svg, pdf)
        
    Returns:
        Path to the generated visualization file
    """
    try:
        # Try to import the rdflib-to-graphviz converter
        from rdflib.tools.rdf2dot import rdf2dot
    except ImportError:
        logger.error("rdflib.tools.rdf2dot not available. Check your rdflib installation.")
        return None

    # Create temporary DOT file
    dot_file = f"{output_file}.dot"
    
    # Convert to DOT format
    with open(dot_file, 'w') as f:
        rdf2dot(graph, f)
    
    # Generate visualization using Graphviz
    output_image = f"{output_file}.{format}"
    
    try:
        # Try to run dot command to generate image
        subprocess.run(["dot", "-T" + format, dot_file, "-o", output_image], 
                       check=True, capture_output=True)
        logger.info(f"Visualization generated at {output_image}")
        return output_image
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating visualization with Graphviz: {e}")
        logger.error(f"Graphviz stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.error("Graphviz dot command not found. Make sure Graphviz is installed.")
        return None
        
def extract_ontology_subset(graph: Graph, limit: int = 100) -> Graph:
    """
    Extract a subset of the ontology for visualization to avoid overwhelming diagrams.
    
    Args:
        graph: Original graph
        limit: Maximum number of triples
        
    Returns:
        Subset graph
    """
    subset = Graph()
    
    # First add namespaces
    for prefix, namespace in graph.namespaces():
        subset.bind(prefix, namespace)
    
    # Add schema triples (classes and properties)
    for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.RDFS.Class)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.Class)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.RDF.Property)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.ObjectProperty)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.DatatypeProperty)):
        subset.add((s, p, o))
        
    # Add domain and range information
    for s, p, o in graph.triples((None, rdflib.RDFS.domain, None)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDFS.range, None)):
        subset.add((s, p, o))
    
    # Add subclass and subproperty relationships
    for s, p, o in graph.triples((None, rdflib.RDFS.subClassOf, None)):
        subset.add((s, p, o))
    
    for s, p, o in graph.triples((None, rdflib.RDFS.subPropertyOf, None)):
        subset.add((s, p, o))
    
    # Add some instance data
    count = 0
    for s, p, o in graph:
        if count >= limit:
            break
        
        # Skip already added triples
        if (s, p, o) in subset:
            continue
            
        subset.add((s, p, o))
        count += 1
    
    return subset