"""
Query functions for the ontology agent.
Used to add query capabilities to the OntologyAgent class.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class OntologyQueryInterface:
    """
    Mixin class that adds ontology query capabilities.
    """
    
    def query_ontology(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the ontology with natural language and return relevant results.
        
        This is a simple implementation that looks for key terms in the query
        and constructs SPARQL queries accordingly.
        
        Args:
            query: The natural language query to process
            
        Returns:
            A list of results matching the query
        """
        query = query.lower()
        results = []
        
        # Load ontology if needed
        if len(self.graph) == 0:
            self._load_ontology()
        
        # Look for specific patterns in the query
        if any(term in query for term in ["ingredient", "ingredients", "made with", "contains", "what's in"]):
            # Query for cocktail ingredients
            if "what cocktails" in query or "which cocktails" in query or "cocktails with" in query:
                # Looking for cocktails with specific ingredient
                for ingredient in ["gin", "vodka", "rum", "tequila", "whiskey", "whisky", "cognac", "brandy"]:
                    if ingredient in query:
                        results.extend(self._query_cocktails_with_ingredient(ingredient))
            else:
                # Looking for ingredients in specific cocktail
                for cocktail in ["martini", "manhattan", "margarita", "mojito", "negroni"]:
                    if cocktail in query:
                        results.extend(self._query_ingredients_for_cocktail(cocktail))
        
        elif any(term in query for term in ["glass", "served in", "container"]):
            # Query for glassware
            for cocktail in ["martini", "manhattan", "margarita", "mojito", "negroni"]:
                if cocktail in query:
                    results.extend(self._query_glass_for_cocktail(cocktail))
        
        # If we found no specific matches, try a general query
        if not results:
            # Use a more general SPARQL query to find anything related to key terms in the query
            results = self._general_ontology_query(query)
        
        logger.info(f"Ontology query '{query}' returned {len(results)} results")
        return results
    
    def _query_cocktails_with_ingredient(self, ingredient: str) -> List[Dict[str, Any]]:
        """Query for cocktails that use a specific ingredient"""
        sparql_query = f"""
        SELECT ?cocktail
        WHERE {{
            ?cocktail rdf:type cocktail:Cocktail .
            ?cocktail property:hasIngredient ?ingredient .
            FILTER(CONTAINS(LCASE(STR(?ingredient)), "{ingredient}"))
        }}
        """
        
        results = []
        try:
            qres = self.graph.query(sparql_query)
            
            for row in qres:
                cocktail_uri = str(row.cocktail)
                cocktail_name = cocktail_uri.split('#')[-1]
                results.append({
                    "type": "cocktail_with_ingredient",
                    "cocktail": cocktail_name,
                    "ingredient": ingredient
                })
        except Exception as e:
            logger.error(f"Error executing SPARQL query for cocktails with ingredient {ingredient}: {e}")
        
        return results
    
    def _query_ingredients_for_cocktail(self, cocktail: str) -> List[Dict[str, Any]]:
        """Query for ingredients in a specific cocktail"""
        sparql_query = f"""
        SELECT ?ingredient
        WHERE {{
            ?c rdf:type cocktail:Cocktail .
            ?c property:hasIngredient ?ingredient .
            FILTER(CONTAINS(LCASE(STR(?c)), "{cocktail}"))
        }}
        """
        
        results = []
        try:
            qres = self.graph.query(sparql_query)
            
            for row in qres:
                ingredient_uri = str(row.ingredient)
                ingredient_name = ingredient_uri.split('#')[-1]
                results.append({
                    "type": "ingredient_in_cocktail",
                    "cocktail": cocktail,
                    "ingredient": ingredient_name
                })
        except Exception as e:
            logger.error(f"Error executing SPARQL query for ingredients in cocktail {cocktail}: {e}")
        
        return results
    
    def _general_ontology_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a general query against the ontology based on key terms"""
        # Extract key terms from the query
        key_terms = [term for term in query.split() if len(term) > 3]
        
        results = []
        
        # Search for these terms in subjects, predicates and objects
        for term in key_terms:
            sparql_query = f"""
            SELECT ?s ?p ?o
            WHERE {{
                ?s ?p ?o .
                FILTER(
                    CONTAINS(LCASE(STR(?s)), "{term}") ||
                    CONTAINS(LCASE(STR(?p)), "{term}") ||
                    CONTAINS(LCASE(STR(?o)), "{term}")
                )
            }}
            LIMIT 20
            """
            
            try:
                qres = self.graph.query(sparql_query)
                
                for row in qres:
                    subject_uri = str(row.s)
                    predicate_uri = str(row.p)
                    object_uri_or_value = str(row.o)
                    
                    subject = subject_uri.split('#')[-1] if '#' in subject_uri else subject_uri
                    predicate = predicate_uri.split('#')[-1] if '#' in predicate_uri else predicate_uri
                    obj = object_uri_or_value.split('#')[-1] if '#' in object_uri_or_value else object_uri_or_value
                    
                    results.append({
                        "type": "general_match",
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj
                    })
            except Exception as e:
                logger.error(f"Error executing general SPARQL query for term {term}: {e}")
        
        return results
    
    def _load_ontology(self):
        """Load the ontology from file if it exists"""
        import os
        try:
            if os.path.exists(self.ontology_file):
                logger.info(f"Loading ontology from {self.ontology_file}")
                self.graph.parse(self.ontology_file, format="turtle")
                logger.info(f"Loaded {len(self.graph)} triples from ontology file")
            else:
                logger.warning(f"Ontology file {self.ontology_file} not found, using empty ontology")
        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
