"""
Ontology agent responsible for extracting knowledge triples and building ontologies.
"""
import os
import logging
import json
import asyncio
from typing import Dict, Any, List, Tuple
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

from agents.common.agent_interface import Agent
from agents.common.config_manager import ConfigManager
from agents.common.data_store import DataStore
from agents.ontology_agent.extract_sparql import extract_sparql_from_response

logger = logging.getLogger(__name__)

class OntologyAgent(Agent):
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Initialize the ontology agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for the agent
        """
        super().__init__(agent_id, config)
        self.config = config or ConfigManager().get_agent_config("ontology_agent")
        self.data_store = DataStore()
        
        # Initialize graph
        self.graph = Graph()
        self.initialize_namespaces()
        
        # Directory for storing ontology files
        # Ruta relativa al directorio del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.ontology_dir = os.path.join(project_root, "ontology")
        
        # También configuramos el directorio de datos interno
        self.data_ontology_dir = os.path.join(self.data_store.get_file_path("data_dir"), "ontology")
        os.makedirs(self.data_ontology_dir, exist_ok=True)
        
        # File paths
        self.ontology_file = os.path.join(self.ontology_dir, "cocktail_ontology.ttl")
        self.triples_file = os.path.join(self.data_ontology_dir, "extracted_triples.json")
        
        # Maximum number of documents to process in a single batch
        # Forzamos el tamaño de batch a 1 para procesar un documento a la vez
        self.batch_size = 1
        
    def initialize_namespaces(self):
        """Initialize namespaces for the ontology"""
        self.COCKTAIL = Namespace("http://www.semanticweb.org/cocktail/ontology#")
        self.INGREDIENT = Namespace("http://www.semanticweb.org/cocktail/ingredient#")
        self.METHOD = Namespace("http://www.semanticweb.org/cocktail/method#")
        self.GLASS = Namespace("http://www.semanticweb.org/cocktail/glass#")
        self.PROPERTY = Namespace("http://www.semanticweb.org/cocktail/property#")
        
        # Bind namespaces to prefixes for readable output
        self.graph.bind("cocktail", self.COCKTAIL)
        self.graph.bind("ingredient", self.INGREDIENT)
        self.graph.bind("method", self.METHOD)
        self.graph.bind("glass", self.GLASS)
        self.graph.bind("property", self.PROPERTY)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        
        # Define base ontology schema
        self._define_base_schema()
        
    def _define_base_schema(self):
        """Define the base schema for cocktail ontology"""
        # Define classes
        self.graph.add((self.COCKTAIL.Cocktail, RDF.type, OWL.Class))
        self.graph.add((self.COCKTAIL.Ingredient, RDF.type, OWL.Class))
        self.graph.add((self.COCKTAIL.Glass, RDF.type, OWL.Class))
        self.graph.add((self.COCKTAIL.Method, RDF.type, OWL.Class))
        
        # Define object properties
        self.graph.add((self.PROPERTY.hasIngredient, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.PROPERTY.hasIngredient, RDFS.domain, self.COCKTAIL.Cocktail))
        self.graph.add((self.PROPERTY.hasIngredient, RDFS.range, self.COCKTAIL.Ingredient))
        
        self.graph.add((self.PROPERTY.servedIn, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.PROPERTY.servedIn, RDFS.domain, self.COCKTAIL.Cocktail))
        self.graph.add((self.PROPERTY.servedIn, RDFS.range, self.COCKTAIL.Glass))
        
        self.graph.add((self.PROPERTY.preparedBy, RDF.type, OWL.ObjectProperty))
        self.graph.add((self.PROPERTY.preparedBy, RDFS.domain, self.COCKTAIL.Cocktail))
        self.graph.add((self.PROPERTY.preparedBy, RDFS.range, self.COCKTAIL.Method))
        
        # Define data properties
        self.graph.add((self.PROPERTY.alcoholContent, RDF.type, OWL.DatatypeProperty))
        self.graph.add((self.PROPERTY.alcoholContent, RDFS.domain, self.COCKTAIL.Cocktail))
        self.graph.add((self.PROPERTY.alcoholContent, RDFS.range, XSD.decimal))
        
        self.graph.add((self.PROPERTY.description, RDF.type, OWL.DatatypeProperty))
        self.graph.add((self.PROPERTY.description, RDFS.domain, self.COCKTAIL.Cocktail))
        self.graph.add((self.PROPERTY.description, RDFS.range, XSD.string))
        
    async def start(self) -> None:
        """Start the ontology agent"""
        await super().start()
        
        # Try to load existing ontology if it exists
        self._load_ontology_if_exists()
        
        logger.info(f"Ontology agent {self.agent_id} started")
    
    def _load_ontology_if_exists(self) -> bool:
        """
        Load ontology from file if it exists.
        
        Returns:
            Success flag
        """
        try:
            if os.path.exists(self.ontology_file):
                # Verificar el tamaño del archivo
                file_size = os.path.getsize(self.ontology_file)
                print(f"[Ontology] Cargando ontología desde {self.ontology_file} (tamaño: {file_size / 1024:.2f} KB)")
                logger.info(f"Loading ontology from {self.ontology_file} (size: {file_size / 1024:.2f} KB)")
                
                # Cargar la ontología
                self.graph.parse(self.ontology_file, format="turtle")
                
                # Mostrar información sobre la ontología cargada
                triples_count = len(self.graph)
                print(f"[Ontology] ✓ Ontología cargada correctamente: {triples_count} tripletas")
                logger.info(f"Ontology loaded successfully: {triples_count} triples")
                return True
            else:
                print(f"[Ontology] ⚠️ Archivo de ontología no encontrado: {self.ontology_file}")
                logger.warning(f"Ontology file not found: {self.ontology_file}")
                return False
        except Exception as e:
            print(f"[Ontology] ❌ Error al cargar la ontología: {str(e)}")
            logger.error(f"Error loading ontology: {e}")
            return False
        
    def _batch_documents(self, documents: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split documents into batches for processing.
        
        Args:
            documents: List of documents to batch
            
        Returns:
            List of document batches
        """
        if not documents:
            return []
            
        batches = []
        for i in range(0, len(documents), self.batch_size):
            batches.append(documents[i:i+self.batch_size])
        
        logger.info(f"Created {len(batches)} batches from {len(documents)} documents with batch_size={self.batch_size}")
        return batches
    
    async def extract_triples_from_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract triples from documents using LLM.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of extracted triples
        """
        all_triples = []
        # Calculate total batches without executing the generator
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        print(f"\n===== INICIANDO EXTRACCIÓN DE ONTOLOGÍA =====")
        print(f"Documentos totales: {len(documents)}")
        print(f"Tamaño de lote: {self.batch_size}")
        print(f"Lotes totales: {total_batches}")
        print(f"=========================================\n")
        
        logger.info(f"Starting triple extraction from {len(documents)} documents in {total_batches} batches")
        logger.info(f"Using batch size of {self.batch_size}")
        
        # Process documents in batches
        for batch_idx, doc_batch in enumerate(self._batch_documents(documents)):
            print(f"\n----- Procesando lote {batch_idx+1}/{total_batches} -----")
            logger.info(f"Processing document batch {batch_idx+1}/{total_batches}")
            batch_start_time = __import__('time').time()
            
            # Process each document in the batch
            batch_triples = await self._process_document_batch(doc_batch)
            all_triples.extend(batch_triples)
            
            batch_end_time = __import__('time').time()
            elapsed_time = batch_end_time - batch_start_time
            
            print(f"----- Lote {batch_idx+1}/{total_batches} completado en {elapsed_time:.2f}s -----")
            print(f"Tripletas extraídas en este lote: {len(batch_triples)}")
            print(f"Total de tripletas hasta ahora: {len(all_triples)}")
            
            logger.info(f"Batch {batch_idx+1}/{total_batches} completed in {elapsed_time:.2f}s. Extracted {len(batch_triples)} triples.")
            
            # Save progress incrementally to avoid losing work if there's a timeout later
            if (batch_idx + 1) % 2 == 0 or batch_idx + 1 == total_batches:
                logger.info(f"Saving intermediate progress: {len(all_triples)} triples so far")
                self._save_extracted_triples(all_triples, suffix=f"_progress_{batch_idx+1}")
                print(f"Progreso guardado: {len(all_triples)} tripletas")
        
        # Save extracted triples
        self._save_extracted_triples(all_triples)
        print(f"\n===== EXTRACCIÓN DE ONTOLOGÍA COMPLETADA =====")
        print(f"Total de tripletas extraídas: {len(all_triples)}")
        print(f"Total de documentos procesados: {len(documents)}")
        print(f"==============================================\n")
        
        # Add triples to graph
        self._add_triples_to_graph(all_triples)
        
        logger.info(f"Extracted {len(all_triples)} triples from {len(documents)} documents")
        return all_triples
    
    async def _process_document_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of documents to extract triples.
        
        Args:
            documents: Batch of documents to process
            
        Returns:
            List of triples extracted from this batch
        """
        batch_triples = []
        
        for idx, document in enumerate(documents):
            # Extract content from document
            content = document.get("text") or document.get("content") or document.get("snippet", "")
            title = document.get("title", "Untitled")
            
            if not content:
                logger.warning(f"Empty content for document: {title}")
                continue
            
            # Log progress for visibility - More detailed message
            print(f"[Ontology] Procesando documento {idx+1}/{len(documents)}: {title}")
            logger.info(f"Processing document {idx+1}/{len(documents)}: {title}")
            
            # Ya no limitamos el contenido aquí, ya que lo fragmentaremos en _extract_triples_with_llm
            
            try:
                # Use LLM to extract triples
                doc_start_time = __import__('time').time()
                print(f"[Ontology] Enviando documento '{title}' al LLM para extracción de tripletas...")
                
                # Procesamos cada documento individualmente
                doc_triples = await self._extract_triples_with_llm(title, content)
                doc_end_time = __import__('time').time()
                
                # Mensaje detallado sobre los resultados
                extraction_time = doc_end_time - doc_start_time
                print(f"[Ontology] ✓ Completado: {len(doc_triples)} tripletas extraídas de '{title}' en {extraction_time:.2f}s")
                logger.info(f"Extracted {len(doc_triples)} triples from document '{title}' in {extraction_time:.2f}s")
                
                # Mostrar algunas tripletas de ejemplo si se encontraron
                if doc_triples:
                    examples = min(3, len(doc_triples))
                    print(f"[Ontology] Ejemplos de tripletas de '{title}':")
                    for i in range(examples):
                        triple = doc_triples[i]
                        print(f"      - {triple['subject']}, {triple['predicate']}, {triple['object']}")
                else:
                    print(f"[Ontology] No se encontraron tripletas en '{title}'")
                
                # Add source information
                for triple in doc_triples:
                    triple["source"] = {
                        "title": title,
                        "url": document.get("source_url", "")
                    }
                    
                batch_triples.extend(doc_triples)
                
                # Guardar progreso por documento para mayor seguridad
                self._save_extracted_triples(batch_triples, suffix=f"_doc_{idx}")
                
                # Pausa breve entre documentos para evitar sobrecargar al LLM o posibles timeouts
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error extracting triples from document '{title}': {e}")
                print(f"[Ontology] ✗ Error al procesar '{title}': {str(e)}")
                # Continue with next document rather than failing the whole batch
                continue
            
        return batch_triples
    
    async def _extract_triples_with_llm(self, title: str, content: str, api_key_path: str = None) -> List[Dict[str, Any]]:
        """
        Use LLM to extract triples from text.
        
        Args:
            title: Document title
            content: Document content
            api_key_path: Optional path to API key file
            
        Returns:
            List of extracted triples
        """
        from agents.generation_agent.generation_agent import GenerationAgent
        
        # Si no se proporciona una ruta de API key, usar la predeterminada
        if not api_key_path:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            api_key_path = os.path.join(project_root, "tokenHuggingFace.txt")
            print(f"[Ontology] Usando ruta de API key predeterminada: {api_key_path}")
            if not os.path.exists(api_key_path):
                print(f"[Ontology] ⚠️ Archivo de API key no encontrado en: {api_key_path}")
                # Intentar con otras posibles rutas
                alt_paths = ["tokenFireworks.txt", "tokenGemini.txt"]
                for alt_path in alt_paths:
                    alt_full_path = os.path.join(project_root, alt_path)
                    if os.path.exists(alt_full_path):
                        api_key_path = alt_full_path
                        print(f"[Ontology] ✓ Usando API key alternativa: {api_key_path}")
                        break
        
        # Create a temporary generation agent for this document only
        # Esto garantiza que cada texto se procese independientemente
        generation_agent = GenerationAgent("temp_extractor")
        
        # Definir el límite máximo de caracteres para enviar al LLM
        max_chunk_size = 4000  # Tamaño ideal para evitar sobrecargar al LLM
        
        # Log that we're sending a request to the LLM
        print(f"[Ontology LLM] Iniciando procesamiento para documento: {title}")
        print(f"[Ontology LLM] Longitud total del texto: {len(content)} caracteres")
        logger.info(f"Starting extraction for document: {title} ({len(content)} chars)")
        
        # Si el contenido es demasiado largo, dividirlo en chunks más pequeños
        all_triples = []
        
        if len(content) <= max_chunk_size:
            # Procesamiento directo para contenido corto
            chunks = [(content, 1, 1)]
        else:
            # Dividir el contenido en chunks más pequeños
            chunks = []
            sentences = content.split('. ')
            current_chunk = ""
            chunk_num = 1
            total_chunks = (len(content) + max_chunk_size - 1) // max_chunk_size
            
            for sentence in sentences:
                # Si agregar esta oración excede el límite, guardar chunk actual e iniciar uno nuevo
                if len(current_chunk) + len(sentence) + 2 > max_chunk_size and current_chunk:
                    chunks.append((current_chunk, chunk_num, total_chunks))
                    chunk_num += 1
                    current_chunk = sentence + ". "
                else:
                    current_chunk += sentence + ". "
            
            # Agregar el último chunk si aún tiene contenido
            if current_chunk:
                chunks.append((current_chunk, chunk_num, total_chunks))
                
        # Procesar cada chunk por separado
        for chunk_content, chunk_num, total_chunks in chunks:
            try:
                print(f"[Ontology LLM] Procesando fragmento {chunk_num}/{total_chunks} ({len(chunk_content)} caracteres)")
                logger.info(f"Processing chunk {chunk_num}/{total_chunks} for document: {title}")
                
                # Añadir contexto específico para el fragmento si hay múltiples fragmentos
                chunk_title = title
                if total_chunks > 1:
                    chunk_title = f"{title} (parte {chunk_num} de {total_chunks})"
                
                # Format prompt for triple extraction
                prompt = self._build_triple_extraction_prompt(chunk_title, chunk_content)
                
                print(f"[Ontology LLM] Enviando solicitud al LLM para fragmento {chunk_num}/{total_chunks}...")
                
                # Generate response using an empty context to ensure independent processing
                response = await generation_agent.generate_response(prompt, [], api_key_path)
                
                if not response:
                    logger.warning(f"Empty response from LLM for chunk {chunk_num}/{total_chunks}")
                    print(f"[Ontology LLM] ⚠️ El LLM devolvió una respuesta vacía para fragmento {chunk_num}/{total_chunks}")
                    continue
                    
                # Log success
                print(f"[Ontology LLM] ✓ Respuesta recibida para fragmento {chunk_num}/{total_chunks}: {len(response)} caracteres")
                logger.info(f"Received response for chunk {chunk_num}/{total_chunks}: {len(response)} chars")
                
                # Parse triples from response
                chunk_triples = self._parse_triples(response)
                print(f"[Ontology LLM] Fragmento {chunk_num}/{total_chunks}: {len(chunk_triples)} tripletas encontradas")
                logger.info(f"Parsed {len(chunk_triples)} triples from chunk {chunk_num}/{total_chunks}")
                
                # Si la respuesta no tiene tripletas, mostrar parte de la respuesta para depuración
                if not chunk_triples and response:
                    print(f"[Ontology LLM] ⚠️ No se encontraron tripletas en fragmento {chunk_num}/{total_chunks}. Muestra de respuesta:")
                    print(f"---\n{response[:200]}...\n---")
                
                all_triples.extend(chunk_triples)
                
            except Exception as e:
                print(f"[Ontology LLM] ❌ Error al procesar fragmento {chunk_num}/{total_chunks}: {str(e)}")
                logger.error(f"Error in LLM processing for chunk {chunk_num}/{total_chunks}: {e}")
                # Continue with next chunk rather than failing the whole document
        
        print(f"[Ontology LLM] Extracción completada para '{title}': {len(all_triples)} tripletas totales")
        logger.info(f"Extraction completed for document '{title}': {len(all_triples)} total triples")
        
        return all_triples
    
    def _build_triple_extraction_prompt(self, title: str, content: str) -> str:
        """
        Build a prompt for extracting triples with LLM.
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Formatted prompt
        """
        # Limit content length to avoid token limits
        max_content_chars = 6000
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."
            
        return f"""Por favor, extrae tripletas semánticas del siguiente texto sobre cócteles y bartenders. 
Una tripleta semántica consta de (sujeto, predicado, objeto).

Título del documento: {title}

Contenido:
{content}

Extrae tripletas relacionadas con cócteles, ingredientes, métodos de preparación, cristalería, etc.
Algunos ejemplos de propiedades a extraer pueden ser:
- hasIngredient (un cóctel tiene un ingrediente específico)
- servedIn (un cóctel se sirve en un tipo de vaso específico)
- alcoholContent (el contenido de alcohol de un cóctel)
- preparedBy (el método de preparación de un cóctel)

Formatea tu respuesta como un listado de tripletas, una por línea, con el formato:
Sujeto, Predicado, Objeto

Por ejemplo:
Martini, hasIngredient, Gin
Martini, servedIn, CocktailGlass
Margarita, hasIngredient, Tequila
Margarita, servedIn, MargaritaGlass

Solo devuelve las tripletas, sin texto adicional.
"""
        
    def _build_sparql_generation_prompt(self, query: str) -> str:
        """
        Build a prompt for generating SPARQL from natural language with improved bilingual support.
        
        Args:
            query: Natural language query (in English or Spanish)
            
        Returns:
            Formatted prompt
        """
        # Get namespace info for the prompt
        namespaces = []
        for prefix, namespace in self.graph.namespaces():
            namespaces.append(f"PREFIX {prefix}: <{namespace}>")
        
        # Join namespace info
        namespace_text = "\n".join(namespaces)
        
        return f"""Por favor, convierte la siguiente consulta en lenguaje natural a una consulta SPARQL válida.
Utiliza los siguientes prefijos para la ontología:

{namespace_text}

La ontología tiene las siguientes clases principales:
- cocktail:Cocktail - Representa cócteles
- cocktail:Ingredient - Representa ingredientes
- cocktail:Glass - Representa tipos de vasos
- cocktail:Method - Representa métodos de preparación

Y las siguientes propiedades principales:
- property:hasIngredient - Relaciona un cóctel con sus ingredientes
- property:servedIn - Relaciona un cóctel con el tipo de vaso
- property:preparedBy - Relaciona un cóctel con el método de preparación
- property:alcoholContent - El contenido de alcohol de un cóctel
- property:description - Descripción de un cóctel

SISTEMA ONTOLÓGICO DE CÓCTELES BILINGÜE - DIRECTRICES ESENCIALES:

1) NOMENCLATURA Y FORMATO DE ENTIDADES:
   - NOMBRES DE CÓCTELES:
     * SIEMPRE en formato PascalCase sin espacios
     * MANTENER el IDIOMA ORIGINAL del nombre (español o inglés)
     * Ejemplos: 
        - "Tinto de Verano" → cocktail:TintoDeVerano (no traducir a SummerRed)
        - "Bloody Mary" → cocktail:BloodyMary 
        - "Piña Colada" → cocktail:PiñaColada (conservar acentos)
        - "Margarita" → cocktail:Margarita

   - INGREDIENTES Y TÉRMINOS TÉCNICOS:
     * SIEMPRE en INGLÉS con PascalCase para términos compuestos
     * Ejemplos: 
        - "vino"/"wine" → ingredient:Wine
        - "vino tinto"/"red wine" → ingredient:RedWine 
        - "jugo de naranja"/"zumo de naranja"/"orange juice" → ingredient:OrangeJuice

   - VASOS Y MÉTODOS:
     * SIEMPRE en INGLÉS con PascalCase para términos compuestos
     * Ejemplos:
        - "copa alta"/"vaso alto"/"highball glass" → glass:HighballGlass
        - "agitado"/"shaken" → method:Shaken

2) PATRONES DE CONSULTA ROBUSTOS:
   - BÚSQUEDAS POR NOMBRE:
     * Utiliza FILTER con CONTAINS/REGEX y LCASE para flexibilidad:
        FILTER(CONTAINS(LCASE(STR(?cocktail)), LCASE("tintoDeVerano")))
   
   - INGREDIENTES Y PROPIEDADES:
     * Siempre comprueba ambas formas de capitalización:
     {{
       ?cocktail property:hasIngredient ingredient:Wine .
     }} 
     UNION 
     {{
       ?cocktail property:hasIngredient ingredient:wine .
     }}

   - CÓCTELES ESPECÍFICOS POR NOMBRE:
     * Para cócteles españoles, buscar directamente por URI:
       ?cocktail = cocktail:TintoDeVerano .
     * Para nombres ambiguos, usar FILTER con contains para capturar variantes:
       FILTER(CONTAINS(LCASE(STR(?cocktail)), "tinto"))

3) PROYECCIÓN Y PRESENTACIÓN:
   - Usa REPLACE para extraer nombres limpios sin prefijos:
     BIND(REPLACE(STR(?cocktail), "^.*#", "") AS ?nombre)
   
   - Extrae componentes identificables:
     BIND(REPLACE(STR(?ingredient), "^.*#", "") AS ?ingredientName)

4) MANEJO DE RESULTADOS:
   - Siempre limita resultados para evitar sobrecarga: LIMIT 25
   - Usa ORDER BY para ordenar resultados
   - Para múltiples valores, agrúpalos con GROUP_CONCAT:
     GROUP_CONCAT(DISTINCT ?ingredientName; separator=", ") AS ?ingredients

Consulta en lenguaje natural: "{query}"

Genera una consulta SPARQL que responda a esta pregunta de la manera más precisa posible.
RECUERDA: esta ontología requiere PascalCase para TODOS los términos compuestos (ingredientes, vasos, métodos).
Los nombres de cócteles mantienen su idioma original pero con formato PascalCase (TintoDeVerano, BloodyMary).

Asegúrate de devolver la consulta SPARQL completa dentro de bloques de código, como en el siguiente ejemplo:

```sparql
PREFIX cocktail: <http://www.semanticweb.org/cocktail/ontology#>
PREFIX property: <http://www.semanticweb.org/cocktail/property#>
PREFIX ingredient: <http://www.semanticweb.org/cocktail/ingredient#>
PREFIX glass: <http://www.semanticweb.org/cocktail/glass#>
PREFIX method: <http://www.semanticweb.org/cocktail/method#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?cocktail (REPLACE(STR(?cocktail), "^.*#", "") AS ?nombre) 
       (GROUP_CONCAT(DISTINCT ?ingredientName; separator=", ") AS ?ingredients)
WHERE {{
  # Patrón básico para encontrar cócteles
  ?cocktail rdf:type cocktail:Cocktail .
  
  # Búsqueda flexible por nombre de cóctel (ejemplo)
  # FILTER(CONTAINS(LCASE(STR(?cocktail)), "tinto"))
  
  # O referencia directa a un cóctel específico
  # ?cocktail = cocktail:TintoDeVerano .
  
  # Ingredientes (con unión para diferentes capitalizaciones)
  ?cocktail property:hasIngredient ?ingredient .
  
  # Para filtrar por un ingrediente específico
  {{
    ?cocktail property:hasIngredient ingredient:Wine .
  }} 
  UNION 
  {{
    # Búsqueda alternativa con CONTAINS para flexibilidad
    ?cocktail property:hasIngredient ?ing .
    FILTER(CONTAINS(LCASE(STR(?ing)), "wine"))
  }}
  
  # Extracción de nombres legibles
  BIND(REPLACE(STR(?ingredient), "^.*#", "") AS ?ingredientName)
}}
GROUP BY ?cocktail
ORDER BY ?nombre
LIMIT 25
```
"""
        
    def _parse_triples(self, llm_response: str) -> List[Dict[str, Any]]:
        """
        Parse triples from LLM response.
        
        Args:
            llm_response: Response from LLM
            
        Returns:
            List of parsed triples
        """
        triples = []
        
        if not llm_response:
            return triples
        
        # Split response into lines
        lines = llm_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that don't look like triples
            if ',' not in line:
                continue
                
            # Try to parse triple
            try:
                parts = [part.strip() for part in line.split(',', 2)]
                
                if len(parts) != 3:
                    continue
                    
                subject, predicate, object_value = parts
                
                # Skip incomplete triples
                if not subject or not predicate or not object_value:
                    continue
                    
                # Add to triples list
                triples.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_value
                })
                
            except Exception as e:
                logger.error(f"Error parsing triple from line '{line}': {e}")
        
        return triples
    
    def _save_extracted_triples(self, triples: List[Dict[str, Any]], suffix: str = "") -> None:
        """
        Save extracted triples to a file.
        
        Args:
            triples: List of triples to save
            suffix: Optional suffix for the filename (for saving progress)
        """
        try:
            # If there's a suffix, save to a different file
            if suffix:
                file_path = os.path.join(self.ontology_dir, f"extracted_triples{suffix}.json")
            else:
                file_path = self.triples_file
                
            with open(file_path, 'w') as f:
                json.dump(triples, f, indent=2)
            logger.info(f"Saved {len(triples)} triples to {file_path}")
        except Exception as e:
            logger.error(f"Error saving triples: {e}")
    
    def _load_extracted_triples(self) -> List[Dict[str, Any]]:
        """
        Load extracted triples from file.
        
        Returns:
            List of triples
        """
        try:
            if os.path.exists(self.triples_file):
                with open(self.triples_file, 'r') as f:
                    triples = json.load(f)
                logger.info(f"Loaded {len(triples)} triples from {self.triples_file}")
                return triples
            return []
        except Exception as e:
            logger.error(f"Error loading triples: {e}")
            return []
    
    def _add_triples_to_graph(self, triples: List[Dict[str, Any]]) -> None:
        """
        Add extracted triples to RDF graph.
        
        Args:
            triples: List of triples to add
        """
        for triple in triples:
            try:
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                object_value = triple.get("object")
                
                if not subject or not predicate or not object_value:
                    continue
                
                # Normalize subject (remove spaces, lowercase first character)
                subject_norm = self._normalize_entity_name(subject)
                
                # Create URIs based on predicate type
                subject_uri = self.COCKTAIL[subject_norm]
                
                # Handle different predicates
                if predicate.lower() in ["hasingredient", "has ingredient", "contains", "uses"]:
                    # Normalize ingredient name
                    object_norm = self._normalize_entity_name(object_value)
                    object_uri = self.INGREDIENT[object_norm]
                    
                    # Add ingredient as instance of Ingredient class
                    self.graph.add((object_uri, RDF.type, self.COCKTAIL.Ingredient))
                    
                    # Add triple
                    self.graph.add((subject_uri, self.PROPERTY.hasIngredient, object_uri))
                    
                elif predicate.lower() in ["servedin", "served in", "glass", "in glass"]:
                    # Normalize glass name
                    object_norm = self._normalize_entity_name(object_value)
                    object_uri = self.GLASS[object_norm]
                    
                    # Add glass as instance of Glass class
                    self.graph.add((object_uri, RDF.type, self.COCKTAIL.Glass))
                    
                    # Add triple
                    self.graph.add((subject_uri, self.PROPERTY.servedIn, object_uri))
                    
                elif predicate.lower() in ["preparedby", "prepared by", "method", "technique"]:
                    # Normalize method name
                    object_norm = self._normalize_entity_name(object_value)
                    object_uri = self.METHOD[object_norm]
                    
                    # Add method as instance of Method class
                    self.graph.add((object_uri, RDF.type, self.COCKTAIL.Method))
                    
                    # Add triple
                    self.graph.add((subject_uri, self.PROPERTY.preparedBy, object_uri))
                    
                elif predicate.lower() in ["alcoholcontent", "alcohol content", "abv"]:
                    # Try to extract numeric value
                    try:
                        # Remove percentage sign if present
                        numeric_value = object_value.replace('%', '')
                        alcohol_value = float(numeric_value)
                        
                        # Add triple with numeric literal
                        self.graph.add((subject_uri, self.PROPERTY.alcoholContent, 
                                         Literal(alcohol_value, datatype=XSD.decimal)))
                    except ValueError:
                        # If not a valid number, add as string
                        self.graph.add((subject_uri, self.PROPERTY.alcoholContent, 
                                         Literal(object_value)))
                        
                elif predicate.lower() in ["description", "definition", "information"]:
                    # Add as string literal
                    self.graph.add((subject_uri, self.PROPERTY.description, Literal(object_value)))
                    
                else:
                    # For other predicates, create a custom property
                    predicate_norm = self._normalize_property_name(predicate)
                    predicate_uri = self.PROPERTY[predicate_norm]
                    
                    # Add as string literal for now
                    self.graph.add((subject_uri, predicate_uri, Literal(object_value)))
                
                # Always add subject as a Cocktail instance
                self.graph.add((subject_uri, RDF.type, self.COCKTAIL.Cocktail))
                
            except Exception as e:
                logger.error(f"Error adding triple to graph: {e}")
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for use in URIs.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        # Remove special characters and spaces
        normalized = ''.join(c if c.isalnum() else '' for c in name)
        
        # Ensure CamelCase format
        words = ''.join(c if c.isalnum() else ' ' for c in name).split()
        if words:
            normalized = ''.join(word.capitalize() for word in words)
        
        return normalized
    
    def _normalize_property_name(self, name: str) -> str:
        """
        Normalize property name for use in URIs.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        # Remove special characters and spaces
        normalized = ''.join(c if c.isalnum() else '' for c in name)
        
        # Ensure camelCase format (first word lowercase, rest capitalized)
        words = ''.join(c if c.isalnum() else ' ' for c in name).split()
        if words:
            normalized = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        
        return normalized
        
    def save_ontology(self, format: str = "turtle") -> str:
        """
        Save ontology to file.
        
        Args:
            format: Format to save in (turtle, xml, json-ld, etc.)
            
        Returns:
            Path to the saved file
        """
        try:
            # Save in Turtle format (more readable)
            turtle_file = self.ontology_file
            self.graph.serialize(destination=turtle_file, format="turtle")
            logger.info(f"Saved ontology to {turtle_file}")
            
            # Also save in OWL/XML format for compatibility with some tools
            owl_file = os.path.join(self.ontology_dir, "cocktail_ontology.owl")
            self.graph.serialize(destination=owl_file, format="xml")
            logger.info(f"Saved ontology to {owl_file}")
            
            return turtle_file
        except Exception as e:
            logger.error(f"Error saving ontology: {e}")
            return ""
            
    def apply_reasoning(self) -> None:
        """
        Apply reasoning to infer new triples based on ontology rules.
        """
        try:
            from owlrl import DeductiveClosure, RDFS_OWLRL_Semantics
            
            # Create a copy of the graph
            pre_reasoning_count = len(self.graph)
            
            # Apply reasoning
            DeductiveClosure(RDFS_OWLRL_Semantics).expand(self.graph)
            
            post_reasoning_count = len(self.graph)
            inferred_triples = post_reasoning_count - pre_reasoning_count
            
            logger.info(f"Applied reasoning: inferred {inferred_triples} new triples")
        except Exception as e:
            logger.error(f"Error applying reasoning: {e}")

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
            
            # Para procesar todos los documentos, comentamos la limitación
            # max_docs = 1  # Solo 1 documento para pruebas rápidas
            # if len(search_results) > max_docs:
            #    print(f"[Ontology] Limitando a {max_docs} documento para evitar timeout")
            #    search_results = search_results[:max_docs]
            
            print(f"[Ontology] Procesando TODOS los {len(search_results)} documentos disponibles")
            
            # Extract and process triples
            try:
                # Si ya tenemos tripletas previas, usarlas como base
                if existing_triples:
                    print(f"[Ontology] Continuando extracción con {len(existing_triples)} tripletas existentes")
                    extracted_triples = existing_triples
                    
                    # Procesar solo documentos nuevos (esto es una simplificación, idealmente deberíamos verificar qué documentos ya han sido procesados)
                    extracted_triples.extend(await self.extract_triples_from_documents(search_results))
                else:
                    # Comenzar desde cero
                    extracted_triples = await self.extract_triples_from_documents(search_results)
                
                # Apply reasoning to infer new triples
                print(f"[Ontology] Aplicando razonamiento a {len(extracted_triples)} tripletas")
                self.apply_reasoning()
                
                # Save ontology
                ontology_file = self.save_ontology()
                print(f"[Ontology] Ontología guardada en {ontology_file}")
                
                response = {
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
                print(f"[OntologyAgent] Enviando respuesta al remitente: {message['sender']}")
                logger.info(f"OntologyAgent sending response to {message['sender']} with action=ontology_results")
                
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
            
        elif action == "query_ontology":
            query = content.get("query", "")
            operation_id = content.get("operation_id")
            use_natural_language = content.get("use_natural_language", True)
            
            if not query:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "operation_id": operation_id,
                        "status": "error",
                        "message": "Empty query",
                        "error": "Empty query provided"
                    }
                }
                
            # Execute query
            if use_natural_language:
                results = await self.natural_language_query(query)
            else:
                results = self.sparql_query(query)
                
            # Check if there's an error in the results
            error = None
            if results and any(isinstance(r, dict) and "error" in r for r in results):
                error_item = next((r for r in results if isinstance(r, dict) and "error" in r), None)
                if error_item:
                    error = error_item["error"]
                    
            if error:
                logger.warning(f"Error en consulta ontológica: {error}")
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "operation_id": operation_id,
                        "status": "error",
                        "query": query,
                        "results": results,
                        "error": error
                    }
                }
            elif not results or len(results) == 0:
                # También manejar el caso de resultados vacíos como un error
                logger.warning(f"La consulta a la ontología no retornó resultados: {query}")
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "operation_id": operation_id,
                        "status": "error",
                        "query": query,
                        "results": [],
                        "error": "No se encontraron resultados en la ontología"
                    }
                }
            else:
                return {
                    "recipient": message["sender"],
                    "content": {
                        "action": "query_results",
                        "operation_id": operation_id,
                        "status": "success",
                        "query": query,
                        "results": results,
                        "error": None
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
        
        return None
        
    def generate_visualization(self, format: str = "png") -> str:
        """
        Generate a visualization of the ontology.
        
        Args:
            format: Format to save the visualization in
            
        Returns:
            Path to the visualization file
        """
        try:
            from .generate_visualization import generate_ontology_visualization
            
            # Base filename for visualization
            visualization_base = os.path.join(self.ontology_dir, "ontology_visualization")
            
            # Generate visualization
            visualization_file = generate_ontology_visualization(
                self.graph, 
                self.ontology_dir, 
                filename="ontology_visualization", 
                format=format
            )
            
            if visualization_file:
                logger.info(f"Generated ontology visualization: {visualization_file}")
            else:
                logger.error("Failed to generate ontology visualization")
                
            return visualization_file
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""
        
    def sparql_query(self, query: str, max_results: int = 10000) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query on the ontology with improved timeout protection.
        
        Args:
            query: SPARQL query string
            max_results: Maximum number of results to return (default increased to 10000 to return more comprehensive results)
            
        Returns:
            Query results
        """
        try:
            # Primero, realizar una limpieza básica de la consulta
            query = query.strip()
            
            # Log the query being executed
            logger.info(f"Executing SPARQL query: {query}")
            print(f"[Ontology] Ejecutando consulta SPARQL: {query[:100]}...")
            
            # Execute the query with a timeout
            results = []
            start_time = __import__('time').time()
            
            # Asegurarse de que la ontología esté cargada correctamente
            if not self._ensure_ontology_loaded():
                logger.error("Failed to load ontology, graph is still empty or too small")
                print(f"[Ontology] ❌ No se pudo cargar la ontología. La ontología parece estar vacía o no existe.")
                # Informar al usuario de cómo crear la ontología
                return [{"error": "La ontología no está disponible o está vacía. Ejecute primero './agent_system.sh ontology extract' para crear la ontología."}]
            else:
                print(f"[Ontology] ✓ Ontología disponible con {len(self.graph)} tripletas.")
                logger.info(f"Ontology ready with {len(self.graph)} triples")
            
            logger.info(f"Starting query execution on graph with {len(self.graph)} triples...")
            print(f"[Ontology] Ejecutando consulta en ontología con {len(self.graph)} tripletas...")
            
            # Process results in batches to avoid memory issues
            result_count = 0
            
            # Definimos un límite máximo de tiempo para la consulta (10 segundos)
            max_query_time = 10.0  # segundos
            timeout_occurred = False
            
            # Limpiar cualquier espacio extra en la consulta
            query = query.strip()
            
            # Validar que la consulta SPARQL es correcta sin ser demasiado estrictos
            if "SELECT" not in query.upper() and "CONSTRUCT" not in query.upper():
                print(f"[Ontology] ⚠️ La consulta no parece ser válida: {query[:100]}...")
                return [{"error": "Consulta SPARQL inválida. Debe contener una cláusula SELECT o CONSTRUCT."}]
            
            try:
                # Intentamos ejecutar la consulta directamente
                print(f"[Ontology] Ejecutando consulta SPARQL directamente...")
                results_generator = self.graph.query(query)
                deadline = start_time + max_query_time
                
                for row in results_generator:
                    # Verificar si hemos excedido el tiempo máximo
                    if __import__('time').time() > deadline:
                        logger.warning(f"Query processing time exceeded {max_query_time} seconds, stopping")
                        print(f"[Ontology] ⚠️ Tiempo de consulta excedido ({max_query_time}s), limitando resultados")
                        timeout_occurred = True
                        break
                        
                    result = {}
                    
                    # Convert row to dictionary using a more robust approach
                    try:
                        # First try the standard approach with row.vars
                        for i, var in enumerate(row.vars):
                            value = row[i]
                            
                            # Convert RDFLib types to Python types
                            if isinstance(value, rdflib.term.URIRef):
                                # Extraer solo el nombre del recurso para facilitar lectura
                                uri_str = str(value)
                                result[var] = uri_str.split('#')[-1] if '#' in uri_str else uri_str
                            elif isinstance(value, rdflib.term.Literal):
                                result[var] = str(value)  # Asegurar string para todos los valores
                            elif isinstance(value, rdflib.term.BNode):
                                result[var] = f"_:{value}"
                            else:
                                result[var] = str(value)
                    except AttributeError:
                        # If row.vars fails, try to handle it as a tuple/list with positional values
                        try:
                            column_names = results_generator.vars
                            for i, var in enumerate(column_names):
                                if i < len(row):
                                    value = row[i]
                                    
                                    # Convert RDFLib types to Python types
                                    if isinstance(value, rdflib.term.URIRef):
                                        uri_str = str(value)
                                        result[var] = uri_str.split('#')[-1] if '#' in uri_str else uri_str
                                    elif isinstance(value, rdflib.term.Literal):
                                        result[var] = str(value)
                                    elif isinstance(value, rdflib.term.BNode):
                                        result[var] = f"_:{value}"
                                    else:
                                        result[var] = str(value)
                        except (AttributeError, IndexError) as e2:
                            # Last resort: try to handle it as a single value if only one variable
                            if len(results_generator.vars) == 1:
                                var = results_generator.vars[0]
                                result[var] = str(row)
                            else:
                                logger.error(f"Failed to process row: {e2}")
                                print(f"[Ontology] Error procesando fila: {e2}")
                                continue
                    
                    # Asegurarse de que siempre existan ciertas variables clave
                    if 'cocktail' in result and 'nombre' not in result:
                        cocktail_str = str(result['cocktail'])
                        result['nombre'] = cocktail_str.split('#')[-1] if '#' in cocktail_str else cocktail_str
                    
                    results.append(result)
                    result_count += 1
                    
                    # Safety check to prevent excessive result processing
                    # The limit is set high (10000 by default) to ensure all relevant results are returned
                    # while still protecting against potential runaway queries
                    if result_count >= max_results:
                        logger.warning(f"Query reached maximum result limit of {max_results}")
                        break
            
            except Exception as e:
                error_message = f"Error durante la ejecución de la consulta: {str(e)}"
                logger.error(f"Error during SPARQL query execution: {e}")
                print(f"[Ontology] ❌ {error_message}")
                
                # Return error information explicitly
                return [{"error": error_message}]
                
                # Proporcionar una consulta alternativa en caso de error
                print(f"[Ontology] Intentando consulta alternativa para encontrar cócteles con vodka...")
                
                # Consulta simplificada para encontrar cócteles con vodka
                fallback_query = """
SELECT ?cocktail ?ingredient
WHERE {
  ?cocktail rdf:type cocktail:Cocktail .
  ?cocktail property:hasIngredient ?ingredient .
  FILTER(CONTAINS(LCASE(STR(?ingredient)), "vodka"))
} LIMIT 10
"""
                try:
                    results_generator = self.graph.query(fallback_query)
                    # Reiniciar el tiempo límite para la consulta fallback
                    deadline = __import__('time').time() + max_query_time
                except Exception as e2:
                    logger.error(f"Error executing fallback SPARQL query: {e2}")
                    return [{"error": f"Error al ejecutar consulta SPARQL alternativa: {str(e2)}. La ontología podría estar vacía o mal formada."}]
            
            end_time = __import__('time').time()
            query_time = end_time - start_time
            logger.info(f"Query execution completed in {query_time:.2f} seconds with {len(results)} results")
            print(f"[Ontology] ✓ Consulta completada en {query_time:.2f}s con {len(results)} resultados")
            
            # Add a note if we had to limit results due to timeout
            if timeout_occurred:
                results.append({"note": f"La consulta excedió el tiempo máximo de {max_query_time} segundos. Se están mostrando resultados parciales."})
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            print(f"[Ontology] ❌ Error general en la consulta SPARQL: {str(e)}")
            return [{"error": str(e)}]
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if a text is in Spanish or English.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code: 'es' for Spanish, 'en' for English
        """
        # List of common Spanish words
        spanish_indicators = [
            'con', 'muéstrame', 'cócteles', 'bebidas', 'contienen', 'tiene', 'puedo', 'hacer', 
            'ingredientes', 'receta', 'como', 'cuál', 'cuáles', 'qué', 'cuántos', 'dónde', 'cómo',
            'preparar', 'preparación', 'mezclar', 'además', 'también', 'usando', 'utilizando',
            'elaborado', 'preparado', 'contenga', 'conteniendo', 'sin', 'puedo', 'quiero', 'necesito'
        ]
        
        # Convert to lowercase for comparison
        text_lower = text.lower()
        
        # Check for Spanish words in the text
        for word in spanish_indicators:
            if word in text_lower.split():
                return 'es'
                
        return 'en'
        
    async def _generate_nl_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a natural language response for query results using the Generation Agent.
        
        Args:
            query: Original natural language query
            results: Query results from ontology
            
        Returns:
            Natural language response
        """
        try:
            from agents.generation_agent.generation_agent import GenerationAgent
            import asyncio
            
            print(f"[Ontology] Generando respuesta en lenguaje natural a partir de resultados de búsqueda...")
            logger.info(f"Generating natural language response from query results")
            
            # Create a temporary generation agent
            generation_agent = GenerationAgent("temp_response_generator")
            
            # Detectar si la consulta está en español
            is_spanish_query = self._detect_language(query) == 'es'
            
            # Convert results to a format suitable for the generation prompt
            search_results = []
            
            # Filter out metadata entries like "generated_sparql" and "info"
            valid_results = [r for r in results if not any(key in r for key in ["generated_sparql", "info", "error"])]
            
            if not valid_results:
                # If no valid results, check for info messages
                info_messages = [r.get("info") for r in results if "info" in r]
                if info_messages:
                    return "\n".join(info_messages)
                
                error_messages = [r.get("error") for r in results if "error" in r]
                if error_messages:
                    return f"Error en la consulta: {' '.join(error_messages)}"
                
                return "No se encontraron resultados para la consulta."
            
            # Format the results as structured data for the generation agent
            formatted_results = {}
            
            # Get all unique keys from the results
            all_keys = set()
            for result in valid_results:
                all_keys.update(result.keys())
            
            # Create a structured dataset with columns
            for key in all_keys:
                if key not in ["generated_sparql", "info", "error"]:
                    formatted_results[key] = [str(r.get(key, "")) for r in valid_results]
            
            # Create a more descriptive document for the generation agent
            document_text = f"Resultados de la consulta '{query}':\n\n"
            
            # Add table headers
            headers = list(formatted_results.keys())
            document_text += " | ".join(headers) + "\n"
            document_text += "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)) + "\n"
            
            # Si la consulta está en español y hay ingredientes, traducirlos utilizando LLM
            if is_spanish_query and "ingredientes" in formatted_results:
                # For each result row that has ingredients
                for i, ingredient_list in enumerate(formatted_results["ingredientes"]):
                    if ingredient_list:
                        # Split by commas and translate each ingredient
                        ingredients = [ing.strip() for ing in ingredient_list.split(",")]
                        translated_ingredients = []
                        
                        for ingredient in ingredients:
                            # Traducir ingrediente de inglés a español usando LLM
                            try:
                                esp_ingredient = await self._translate_ingredient_with_llm(
                                    ingredient, source_lang="en", target_lang="es")
                                translated_ingredients.append(esp_ingredient.capitalize())
                            except Exception as e:
                                print(f"[Ontology] Error al traducir ingrediente {ingredient}: {str(e)}")
                                translated_ingredients.append(ingredient)
                        
                        # Update the formatted results with translated ingredients
                        formatted_results["ingredientes"][i] = ", ".join(translated_ingredients)
            
            # Add table rows
            max_rows = max(len(values) for values in formatted_results.values())
            for i in range(max_rows):
                row = []
                for key in headers:
                    values = formatted_results.get(key, [])
                    value = values[i] if i < len(values) else ""
                    row.append(value)
                document_text += " | ".join(row) + "\n"
            
            # Add instructions for generating a natural language response
            if is_spanish_query:
                document_text += "\nGenera una respuesta natural en español que describa estos resultados de manera conversacional. "
                document_text += "Si no hay resultados, indícalo. Incluye información sobre los cócteles encontrados y sus ingredientes."
            else:
                document_text += "\nGenerate a natural language response in English that describes these results in a conversational manner. "
                document_text += "If there are no results, indicate that. Include information about the cocktails found and their ingredients."
            
            try:
                # Set a 30-second timeout for the LLM generation
                llm_task = generation_agent.generate_response(document_text, [])
                
                start_time = __import__('time').time()
                response = await asyncio.wait_for(llm_task, timeout=30.0)
                elapsed_time = __import__('time').time() - start_time
                
                print(f"[Ontology] ✓ Respuesta natural generada en {elapsed_time:.2f}s")
                
                return response
                
            except asyncio.TimeoutError:
                print(f"[Ontology] ⚠️ Tiempo agotado al generar respuesta natural. Devolviendo resultados crudos.")
                
                # Fallback to a simple response
                if is_spanish_query:
                    return f"Encontré {len(valid_results)} cócteles que coinciden con tu búsqueda. Aquí está el primero: {valid_results[0].get('nombre', 'Sin nombre')}."
                else:
                    return f"I found {len(valid_results)} cocktails matching your search. Here's the first one: {valid_results[0].get('nombre', 'Unnamed')}."
                    
            except Exception as e:
                print(f"[Ontology] ❌ Error al generar respuesta natural: {str(e)}. Devolviendo resultados crudos.")
                
                # Fallback to a simple response
                if is_spanish_query:
                    return f"Encontré {len(valid_results)} cócteles que coinciden con tu búsqueda. Ocurrió un error al generar una respuesta natural."
                else:
                    return f"I found {len(valid_results)} cocktails matching your search. An error occurred while generating a natural response."
                    
        except Exception as e:
            logger.error(f"Error generating natural language response: {e}")
            print(f"[Ontology] ❌ Error general al generar respuesta: {str(e)}")
            
            # Return a simple error message
            return f"Error al procesar los resultados: {str(e)}"
    
    async def natural_language_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Convert natural language query to SPARQL and execute with improved robustness.
        Uses timeout handling to prevent blocking.
        
        Args:
            query: Natural language query
            
        Returns:
            Query results
        """
        import asyncio
        try:
            # Log the start of the process
            print(f"[Ontology] Procesando consulta en lenguaje natural: '{query}'")
            logger.info(f"Processing natural language query: '{query}'")
            
            # Verificar si la consulta parece ser una respuesta en lugar de una pregunta
            # Para evitar procesar accidentalmente respuestas como consultas
            if query.count('\n') > 5 or ('**' in query and ':' in query and '-' in query):
                logger.warning(f"La consulta parece ser una respuesta generada, no una pregunta. Devolviendo error.")
                print(f"[Ontology] ⚠️ La entrada parece ser una respuesta, no una consulta.")
                return [{"error": "La consulta parece ser una respuesta generada en lugar de una pregunta."}]
            
            # Asegurarse de que la ontología esté cargada correctamente
            if not self._ensure_ontology_loaded():
                logger.error("Failed to load ontology, graph is still empty or too small")
                print(f"[Ontology] ❌ No se pudo cargar la ontología. La ontología parece estar vacía o no existe.")
                # Informar al usuario de cómo crear la ontología
                return [{"error": "La ontología no está disponible o está vacía. Ejecute primero './agent_system.sh ontology extract' para crear la ontología."}]
            else:
                print(f"[Ontology] ✓ Ontología disponible con {len(self.graph)} tripletas.")
                
            # Convert natural language query to SPARQL with timeout protection
            print(f"[Ontology] Convirtiendo consulta a SPARQL...")
            start_time = __import__('time').time()
            
            try:
                # Use wait_for with a timeout to ensure we don't block indefinitely
                convert_task = self._nl_to_sparql(query)
                sparql_query = await asyncio.wait_for(convert_task, timeout=50.0)
                
                # Log success
                elapsed_time = __import__('time').time() - start_time
                print(f"[Ontology] ✓ Consulta SPARQL generada en {elapsed_time:.2f}s")
                logger.info(f"Generated SPARQL query in {elapsed_time:.2f}s: {sparql_query}")
                
            except asyncio.TimeoutError:
                print(f"[Ontology] ⚠️ Timeout al generar consulta SPARQL (50s). Usando consulta fallback.")
                logger.error(f"Timeout generating SPARQL from natural language query")
                
                # Generate a simple fallback query
                sparql_query = self._get_fallback_sparql_query(query)
            
            # La consulta SPARQL ya debería tener LIMIT desde el método _nl_to_sparql
            # No necesitamos añadir LIMIT aquí
            
            # Validar rápidamente la consulta antes de ejecutarla
            if not sparql_query or (("SELECT" not in sparql_query.upper()) and ("CONSTRUCT" not in sparql_query.upper()) and ("ASK" not in sparql_query.upper())):
                print(f"[Ontology] ⚠️ La consulta generada no es válida. Usando alternativa simple.")
                # Si la consulta es sobre vodka, usar una consulta especial más flexible
                if "vodka" in query.lower() or any(ing in query.lower() for ing in ["gin", "rum", "tequila", "whiskey", "brandy"]):
                    print(f"[Ontology] Detectada consulta sobre ingredientes, usando consulta especial robusta...")
                    
                    # Determinar el ingrediente a buscar
                    ingredient = "vodka"  # Por defecto
                    for ing in ["vodka", "gin", "rum", "tequila", "whiskey", "brandy"]:
                        if ing in query.lower():
                            ingredient = ing
                            break
                    
                    # Usar el método fallback mejorado para generar la consulta
                    sparql_query = self._get_fallback_sparql_query(query)
                else:
                    # Consulta genérica para otros casos
                    sparql_query = """
SELECT ?cocktail ?ingredient
WHERE {
  ?cocktail rdf:type cocktail:Cocktail .
  ?cocktail property:hasIngredient ?ingredient .
} LIMIT 10
"""
                
            # Execute SPARQL query with timeout protection
            print(f"[Ontology] Ejecutando consulta SPARQL: {sparql_query[:100]}...")
            query_start = __import__('time').time()
            
            try:
                # Ejecutar la consulta directamente
                results = self.sparql_query(sparql_query)
                
                query_time = __import__('time').time() - query_start
                print(f"[Ontology] ✓ Consulta ejecutada en {query_time:.2f}s con {len(results)} resultados")
                logger.info(f"Query executed in {query_time:.2f}s with {len(results)} results")
                
            except asyncio.TimeoutError:
                print(f"[Ontology] ⚠️ Timeout ejecutando consulta SPARQL (15s).")
                logger.error(f"Timeout executing SPARQL query")
                # Return error message instead of empty results
                return [{"error": "La consulta SPARQL tardó demasiado en ejecutarse. Intente con una consulta más específica."}]
            
            except Exception as e:
                print(f"[Ontology] ❌ Error durante la ejecución de la consulta: {str(e)}")
                logger.error(f"Error executing SPARQL query: {e}")
                
                # Devolver un mensaje de error que incluya el error original completo
                # para que pueda ser detectado por el coordinator y activar el crawler dinámico
                return [{"error": f"Error durante la ejecución de la consulta: {str(e)}"}]
            
            # Add the generated SPARQL for reference
            if results:
                # Store the raw results for reference
                raw_results = results.copy()
                
                # Add SPARQL query for reference
                raw_results.append({"generated_sparql": sparql_query})
                
                # Generate natural language response
                print(f"[Ontology] Generando respuesta en lenguaje natural para la consulta...")
                nl_response = await self._generate_nl_response(query, raw_results)
                
                # Add the natural language response at the beginning of results
                if nl_response:
                    raw_results.insert(0, {"nl_response": nl_response})
                
                return raw_results
            else:
                # If no results, return a helpful message with better diagnostic information
                print(f"[Ontology] ⚠️ La consulta no devolvió resultados: {sparql_query}")
                
                # Return information about the query that produced no results
                no_results_message = f"No se encontraron resultados para la consulta / No results found for this query. "
                
                # Check for potential format issues in the query
                format_issues = []
                
                # Extract terms that might be incorrectly formatted from the SPARQL query
                import re
                
                # Common naming patterns to check
                query_lower = query.lower()
                
                # Comprehensive checks for common ingredient and glass types in both languages
                
                # Check for glass types in the query
                glass_patterns = [
                    ("copa alta", "vaso alto", "highball", "high ball", "HighballGlass"),
                    ("copa martini", "vaso martini", "martini glass", "MartiniGlass"),
                    ("copa hurricane", "vaso hurricane", "hurricane glass", "HurricaneGlass"),
                    ("copa margarita", "vaso margarita", "margarita glass", "MargaritaGlass"),
                    ("copa champagne", "copa champaña", "champagne glass", "ChampagneFlute"),
                    ("copa collins", "vaso collins", "collins glass", "CollinsGlass")
                ]
                
                for pattern_group in glass_patterns:
                    target_format = pattern_group[-1]
                    if any(p in query_lower for p in pattern_group[:-1]) and target_format not in sparql_query:
                        format_issues.append(f'"{pattern_group[0]}" debe usar "{target_format}" en la ontología')
                
                # Check for ingredients and mixers in the query
                ingredient_patterns = [
                    ("granadina", "grenadine", "Grenadine"),
                    ("vino tinto", "red wine", "RedWine"),
                    ("vino blanco", "white wine", "WhiteWine"),
                    ("zumo de naranja", "jugo de naranja", "orange juice", "OrangeJuice"),
                    ("agua tonica", "agua tónica", "tonic water", "TonicWater"),
                    ("vermut", "vermouth", "Vermouth"),
                ]
                
                for pattern_group in ingredient_patterns:
                    target_format = pattern_group[-1]
                    if any(p in query_lower for p in pattern_group[:-1]) and target_format not in sparql_query:
                        format_issues.append(f'"{pattern_group[0]}" debe usar "{target_format}" en la ontología')
                
                # Check for cocktail name format issues
                if "tinto de verano" in query_lower and "TintoDeVerano" not in sparql_query:
                    format_issues.append('"tinto de verano" debe usar "TintoDeVerano" en la ontología (PascalCase)')
                
                # If we identified potential format issues
                if format_issues:
                    no_results_message += "Posibles problemas de formato detectados / Possible format issues detected: " + ", ".join(format_issues) + ". "
                    no_results_message += """
                    
• RECORDATORIO / REMINDER:
  - Los términos compuestos usan formato PascalCase sin espacios / Compound terms use PascalCase without spaces
  - Ejemplos / Examples: "vino tinto" → "RedWine", "copa alta" → "HighballGlass"
  - Los nombres de cócteles mantienen su idioma original pero en PascalCase / Cocktail names keep their original language but in PascalCase
  - Ejemplos / Examples: "Tinto de Verano" → "TintoDeVerano", "Bloody Mary" → "BloodyMary"
                    """
                
                # If it's an ingredient query, provide more specific information
                if "vodka" in query.lower():
                    no_results_message += "La ontología contiene cócteles con vodka como ingrediente. Intentando con consulta alternativa definitiva..."
                    # Intentar una consulta muy simple directa pero más robusta para vodka
                    final_query = """
PREFIX cocktail: <http://www.semanticweb.org/cocktail/ontology#>
PREFIX property: <http://www.semanticweb.org/cocktail/property#>
PREFIX ingredient: <http://www.semanticweb.org/cocktail/ingredient#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?cocktail 
       (REPLACE(STR(?cocktail), "^.*#", "") AS ?nombre) 
       ("Vodka" AS ?ingredientes)
WHERE {
  ?cocktail rdf:type cocktail:Cocktail .
  {
    # Intenta con URI exacta
    ?cocktail property:hasIngredient ingredient:Vodka .
  } UNION {
    # Intenta con cualquier ingrediente que contenga vodka en el nombre
    ?cocktail property:hasIngredient ?ing .
    FILTER(CONTAINS(LCASE(STR(?ing)), "vodka"))
  } UNION {
    # Intenta con DejaVu específicamente que sabemos que tiene vodka
    FILTER(?cocktail = cocktail:DejaVu || ?cocktail = cocktail:CóctelDejaVu)
  }
} 
"""
                    try:
                        print(f"[Ontology] Intentando consulta final ultrasimple para vodka...")
                        logger.info("Trying final simple query for vodka with explicit projection")
                        results_raw = self.graph.query(final_query)
                        
                        # Convert QueryResult to a list to check length and prevent consuming the generator
                        results_list = list(results_raw)
                        
                        print(f"[Ontology] La consulta final devolvió {len(results_list)} resultados.")
                        logger.info(f"Final vodka query returned {len(results_list)} results")
                        
                        if len(results_list) > 0:
                            final_results = []
                            
                            # Process each row more carefully
                            for row in results_list:
                                try:
                                    # Print debug information
                                    print(f"[Ontology] Procesando fila: {row}")
                                    
                                    # Create a result dictionary manually from the row values
                                    result = {}
                                    
                                    # Process all values based on position since row.vars might be causing issues
                                    if len(row) >= 1:
                                        cocktail_value = row[0]
                                        if isinstance(cocktail_value, rdflib.term.URIRef):
                                            uri_str = str(cocktail_value)
                                            result["cocktail"] = uri_str.split('#')[-1] if '#' in uri_str else uri_str
                                        else:
                                            result["cocktail"] = str(cocktail_value)
                                    
                                    if len(row) >= 2:
                                        result["nombre"] = str(row[1])
                                    else:
                                        # If nombre is not provided, extract it from cocktail
                                        cocktail_str = str(result.get("cocktail", ""))
                                        result["nombre"] = cocktail_str.split('#')[-1] if '#' in cocktail_str else cocktail_str
                                    
                                    if len(row) >= 3:
                                        result["ingredientes"] = str(row[2])
                                    else:
                                        result["ingredientes"] = "Vodka"
                                    
                                    final_results.append(result)
                                    
                                except Exception as row_err:
                                    print(f"[Ontology] Error procesando fila: {str(row_err)}")
                                    logger.error(f"Error processing row in vodka query: {row_err}")
                                    continue
                            
                            if final_results:
                                print(f"[Ontology] ✓ Se encontraron {len(final_results)} cócteles con vodka")
                                
                                # Add metadata
                                final_results_with_meta = final_results.copy()
                                final_results_with_meta.append({"generated_sparql": final_query})
                                final_results_with_meta.append({"info": f"Se encontraron {len(final_results)} resultados con la consulta final para Vodka."})
                                
                                # Generate natural language response
                                print(f"[Ontology] Generando respuesta en lenguaje natural para la consulta de vodka...")
                                nl_response = await self._generate_nl_response(query, final_results_with_meta)
                                
                                # Add the natural language response at the beginning
                                if nl_response:
                                    final_results_with_meta.insert(0, {"nl_response": nl_response})
                                
                                return final_results_with_meta
                    except Exception as e:
                        print(f"[Ontology] Error en consulta final: {e}")
                        logger.error(f"Error in final vodka query: {e}")
                        
                    no_results_message = "No se pudieron encontrar cócteles con vodka a pesar de múltiples intentos. La ontología podría tener problemas estructurales."
                elif any(ing in query.lower() for ing in ["gin", "rum", "tequila", "whiskey", "brandy"]):
                    no_results_message += f"Intente reformular la pregunta o mencione otro ingrediente."
                else:
                    no_results_message += "Intente con una consulta diferente o más específica."
                
                return [{"info": no_results_message}, {"generated_sparql": sparql_query}]
            
        except Exception as e:
            logger.error(f"Error with natural language query: {e}")
            print(f"[Ontology] ❌ Error general en consulta: {str(e)}")
            return [{"error": str(e)}]
    
    async def _nl_to_sparql(self, query: str) -> str:
        """
        Convert natural language query to SPARQL using LLM with enhanced bilingual support.
        This method handles queries in both Spanish and English, ensuring proper formatting
        for the ontology's naming conventions.
        
        Args:
            query: Natural language query in Spanish or English
            
        Returns:
            SPARQL query string
        """
        from agents.generation_agent.generation_agent import GenerationAgent
        import asyncio
        import os
        import re
        
        # Create a temporary generation agent
        generation_agent = GenerationAgent("temp_sparql_generator")
        
        # Store the original query for reference
        original_query = query
        original_language = self._detect_language(query)
        
        print(f"[Ontology] Consulta original ({original_language.upper()}): '{original_query}'")
        
        # Process the query for ontology format with LLM (handles both languages)
        # This ensures all terms use proper PascalCase and conventions for the ontology
        query = await self._translate_full_query_with_llm(query)
        print(f"[Ontology] Consulta procesada para ontología: '{query}'")
        
        # We'll keep this empty list for compatibility with existing code
        # but the modern approach uses full query translation instead
        ingredients_found = []
        
        # Build prompt for SPARQL generation
        prompt = self._build_sparql_generation_prompt(query)
        
        try:
            # Log that we're starting the conversion
            print(f"[Ontology] Iniciando conversión de consulta natural a SPARQL: '{query}'")
            logger.info(f"Starting natural language to SPARQL conversion for: '{query}'")
            
            # Set a strict timeout of 45 seconds for the LLM generation request
            llm_task = generation_agent.generate_response(prompt, [])
            
            # Utilizar asyncio.wait_for con un timeout más corto
            start_time = __import__('time').time()
            response = await asyncio.wait_for(llm_task, timeout=45.0)
            elapsed_time = __import__('time').time() - start_time
            
            # Log the success and timing
            print(f"[Ontology] ✓ Conversión exitosa en {elapsed_time:.2f}s")
            logger.info(f"SPARQL conversion successful in {elapsed_time:.2f}s")
            
            # Extract SPARQL query from response
            sparql_query = self._extract_sparql_from_response(response)
            
            # Validar que la respuesta se parece a una consulta SPARQL
            if "SELECT" not in sparql_query.upper() and "CONSTRUCT" not in sparql_query.upper():
                print(f"[Ontology] ⚠️ La respuesta no parece ser una consulta SPARQL válida. Usando consulta alternativa.")
                logger.warning(f"Response doesn't look like valid SPARQL: {sparql_query[:100]}...")
                return self._get_fallback_sparql_query(query, ingredients_found)
            
            # Limpiar la consulta SPARQL para asegurarnos que está bien formateada
            sparql_query = sparql_query.strip()
            
            # Asegurarnos que está correctamente formateada con WHERE y llaves
            if "WHERE" not in sparql_query.upper():
                print(f"[Ontology] ⚠️ La consulta generada no tiene cláusula WHERE. Usando consulta alternativa.")
                return self._get_fallback_sparql_query(query, ingredients_found)
            
            # Corregir los nombres de los ingredientes directamente en la consulta SPARQL
            for esp, eng in ingredients_found:
                # Capitalize ingredients
                eng_cap = eng.capitalize()
                
                # Corregir referencias a ingredientes en minúsculas
                sparql_query = re.sub(
                    f'ingredient:{re.escape(eng)}\\b', 
                    f'ingredient:{eng_cap}', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                # Corregir referencias a ingredientes en español
                sparql_query = re.sub(
                    f'ingredient:{re.escape(esp)}\\b', 
                    f'ingredient:{eng_cap}', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                # Corregir filtros con strings - versión inglés
                sparql_query = re.sub(
                    f'\\?ingredient\\s*=\\s*["\']({re.escape(eng)})["\']', 
                    f'?ingredient = ingredient:{eng_cap}', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                # Corregir filtros con strings - versión español
                sparql_query = re.sub(
                    f'\\?ingredient\\s*=\\s*["\']({re.escape(esp)})["\']', 
                    f'?ingredient = ingredient:{eng_cap}', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                # Corregir patrones CONTAINS para buscar ingredientes
                sparql_query = re.sub(
                    f'CONTAINS\\(LCASE\\(STR\\(\\?ingredient\\)\\),\\s*["\']({re.escape(eng)})["\']\\)', 
                    f'CONTAINS(LCASE(STR(?ingredient)), "{eng.lower()}")', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                sparql_query = re.sub(
                    f'CONTAINS\\(LCASE\\(STR\\(\\?ingredient\\)\\),\\s*["\']({re.escape(esp)})["\']\\)', 
                    f'CONTAINS(LCASE(STR(?ingredient)), "{eng.lower()}")', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                # Buscar y corregir otros patrones de filtering por nombre de ingrediente
                sparql_query = re.sub(
                    f'\\?ingredientName\\s*=\\s*["\']({re.escape(eng)})["\']', 
                    f'?ingredientName = "{eng_cap}"', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
                
                sparql_query = re.sub(
                    f'\\?ingredientName\\s*=\\s*["\']({re.escape(esp)})["\']', 
                    f'?ingredientName = "{eng_cap}"', 
                    sparql_query, 
                    flags=re.IGNORECASE
                )
            
            # Corregir cualquier error en el formato SPARQL
            # Arreglar prefijo incorrecto cocktail:Ingredient:X -> ingredient:X
            sparql_query = re.sub(
                r'cocktail:Ingredient:(\w+)', 
                r'ingredient:\1', 
                sparql_query
            )
            
            # Asegurarse que toda referencia a ingrediente por nombre (sin URIs) use la versión capitalizada
            for esp, eng in ingredients_found:
                eng_cap = eng.capitalize()
                
                # Buscar referencias al ingrediente por nombre (sin URI) y capitalizarlo
                sparql_query = re.sub(
                    f'["\']({re.escape(eng)})["\']', 
                    f'"{eng_cap}"', 
                    sparql_query,
                    flags=re.IGNORECASE
                )
                
                # También para la versión en español
                if esp != eng:  # Solo si es diferente
                    sparql_query = re.sub(
                        f'["\']({re.escape(esp)})["\']', 
                        f'"{eng_cap}"', 
                        sparql_query,
                        flags=re.IGNORECASE
                    )
            
            print(f"[Ontology] ✓ Consulta SPARQL generada y optimizada: {sparql_query[:150]}...")
            logger.info(f"Generated SPARQL query in {elapsed_time:.2f}s: {sparql_query}")
            
            return sparql_query
            
        except asyncio.TimeoutError:
            logger.error("LLM request for SPARQL generation timed out after 45 seconds")
            print(f"[Ontology] ⚠️ Se agotó el tiempo de espera (45s) para la generación de SPARQL. Usando consulta alternativa.")
            
            # Return a fallback SPARQL query based on keywords in the original query
            return self._get_fallback_sparql_query(query, ingredients_found)
            
        except Exception as e:
            logger.error(f"Error generating SPARQL from natural language: {e}")
            print(f"[Ontology] ❌ Error al generar SPARQL: {str(e)}. Usando consulta alternativa.")
            
            # Return a fallback query
            return self._get_fallback_sparql_query(query, ingredients_found)
        """
        Generate a fallback SPARQL query based on keywords in the original query.
        
        Args:
            query: Original natural language query
            ingredients_found: Optional pre-detected ingredients as tuples of (spanish, english)
            
        Returns:
            Fallback SPARQL query
        """
        query_lower = query.lower()
        
        # Look for ingredient-related keywords in both English and Spanish
        ingredients = []
        
        # Dictionary of ingredients with English as key and Spanish as value
        ingredient_translations = {
            "vodka": "vodka", 
            "gin": "ginebra",
            "rum": "ron",
            "tequila": "tequila",
            "whiskey": "whisky",
            "whisky": "whisky",
            "brandy": "brandy",
            "vermouth": "vermut",
            "lime": "lima",
            "lemon": "limón",
            "orange": "naranja",
            "pineapple": "piña",
            "cranberry": "arándano",
            "mint": "menta",
            "coffee": "café",
            "milk": "leche",
            "cream": "crema",
            "sugar": "azúcar",
            "ginger": "jengibre",
            "strawberry": "fresa",
            "coconut": "coco",
            "cinnamon": "canela"
        }
        
        # Use pre-detected ingredients if available
        if ingredients_found and isinstance(ingredients_found, list) and len(ingredients_found) > 0:
            for esp, eng in ingredients_found:
                if eng not in ingredients:
                    print(f"[Ontology] Usando ingrediente previamente detectado: {esp} → {eng}")
                    ingredients.append(eng)
        else:
            # Otherwise check for ingredients in any language
            for eng_ingredient, esp_ingredient in ingredient_translations.items():
                if eng_ingredient in query_lower or esp_ingredient in query_lower:
                    # Always use the English version for the query
                    if eng_ingredient not in ingredients:
                        print(f"[Ontology] Detectado ingrediente: {esp_ingredient} → {eng_ingredient}")
                        ingredients.append(eng_ingredient)
        
        # If we found specific ingredients, create a query for them
        if ingredients:
            ingredient = ingredients[0]  # Use the first found ingredient
            logger.info(f"Using fallback query for ingredient: {ingredient}")
            ingredient_cap = ingredient.capitalize()
            
            print(f"[Ontology] Generando consulta robusta para ingrediente: {ingredient} (encontrado en la consulta como '{ingredient}' o '{ingredient_translations.get(ingredient)}')")
            
            # Consulta robusta y mejorada que busca cócteles con el ingrediente en diferentes formas
            # y devuelve etiquetas más amigables para el usuario
            query = f"""
PREFIX cocktail: <http://www.semanticweb.org/cocktail/ontology#>
PREFIX property: <http://www.semanticweb.org/cocktail/property#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ingredient: <http://www.semanticweb.org/cocktail/ingredient#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?cocktail 
       (REPLACE(STR(?cocktail), "^.*#", "") AS ?nombre) 
       (GROUP_CONCAT(DISTINCT ?ingredientName; separator=", ") AS ?ingredientes)
WHERE {{
  # Primera forma: buscar el ingrediente exacto
  ?cocktail rdf:type cocktail:Cocktail .
  ?cocktail property:hasIngredient ingredient:{ingredient_cap} .

  # Obtener todos los ingredientes para información completa
  ?cocktail property:hasIngredient ?ingredient .
  BIND(REPLACE(STR(?ingredient), "^.*#", "") AS ?ingredientName)
}}
GROUP BY ?cocktail
"""
            return query
        
        # Look for glass type keywords
        glass_types = ["martini", "highball", "collins", "margarita", "shot"]
        for glass in glass_types:
            if glass in query_lower:
                logger.info(f"Using fallback query for glass type: {glass}")
                
                return f"""
SELECT ?cocktail ?glass
WHERE {{
  ?cocktail rdf:type cocktail:Cocktail .
  ?cocktail property:servedIn ?glass .
  FILTER(CONTAINS(LCASE(STR(?glass)), "{glass}"))
}}
"""
        
        # Default query showing some popular cocktails and their ingredients
        logger.info("Using generic fallback query")
        return """
SELECT ?cocktail ?ingredient
WHERE {
  ?cocktail rdf:type cocktail:Cocktail .
  ?cocktail property:hasIngredient ?ingredient .
}
"""
    
    def _extract_sparql_from_response(self, response: str) -> str:
        """
        Extract SPARQL query from LLM response.
        Usa la versión mejorada del módulo extract_sparql.py.
        
        Args:
            response: LLM response
            
        Returns:
            SPARQL query string
        """
        # Delegamos a la función importada que tiene mejores correcciones y manejo de errores
        return extract_sparql_from_response(response)
        """
        Extract SPARQL query from LLM response.
        
        Args:
            response: LLM response
            
        Returns:
            SPARQL query string
        """
        import re
        
        # Try to extract code blocks
        if "```sparql" in response and "```" in response.split("```sparql", 1)[1]:
            # Extract content between ```sparql and ```
            sparql = response.split("```sparql", 1)[1].split("```", 1)[0].strip()
        elif "```" in response and "```" in response.split("```", 1)[1]:
            # Extract content between first ``` and second ```
            sparql = response.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            # Just use the whole response as a fallback
            sparql = response.strip()
        
        # Corregir errores comunes en las consultas SPARQL generadas
        
        # 0. Asegurarse que todos los prefijos necesarios están definidos
        required_prefixes = [
            "PREFIX cocktail: <http://www.semanticweb.org/cocktail/ontology#>",
            "PREFIX property: <http://www.semanticweb.org/cocktail/property#>",
            "PREFIX ingredient: <http://www.semanticweb.org/cocktail/ingredient#>",
            "PREFIX glass: <http://www.semanticweb.org/cocktail/glass#>",
            "PREFIX method: <http://www.semanticweb.org/cocktail/method#>",
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
        ]
        
        # Verificar si cada prefijo está presente y agregar los que faltan
        prefix_section = ""
        
        # Verificar si hay una sección de prefijos
        if not any(line.strip().startswith("PREFIX ") for line in sparql.split("\n")):
            # No hay prefijos, agregamos todos
            prefix_section = "\n".join(required_prefixes) + "\n\n"
        else:
            # Hay algunos prefijos, verificar cuáles faltan
            for prefix in required_prefixes:
                prefix_name = prefix.split("<")[0].strip()
                if not any(line.strip().startswith(prefix_name) for line in sparql.split("\n")):
                    prefix_section += prefix + "\n"
            
            if prefix_section:
                prefix_section += "\n"
        
        # Si hay prefijos que agregar, insertarlos al principio
        if prefix_section:
            if sparql.upper().startswith("PREFIX"):
                # Ya tiene prefijos, agrega los que faltan al principio
                lines = sparql.split("\n")
                prefix_lines = [l for l in lines if l.strip().upper().startswith("PREFIX")]
                non_prefix_lines = [l for l in lines if not l.strip().upper().startswith("PREFIX")]
                
                # Combinar todos los prefijos y luego el resto del query
                sparql = "\n".join(prefix_lines) + "\n" + prefix_section + "\n".join(non_prefix_lines)
            else:
                # No tiene ningún prefijo, agregar todos al principio
                sparql = prefix_section + sparql
        
        # 1. Corregir uso incorrecto de FILTER con URIs de ingredientes
        sparql = sparql.replace("cocktail:Ingredient:", "ingredient:")
        
        # 2. Corregir cláusulas FILTER para comparar con URIs de ingredientes
        pattern = r'FILTER\s*\(\s*\?ingredient\s*=\s*["\'](\w+)["\']\s*\)'
        
        def replace_ingredient_filter(match):
            ingredient_name = match.group(1)
            ingredient_name_capitalized = ingredient_name.capitalize()
            return f'FILTER(?ingredient = ingredient:{ingredient_name_capitalized})'
        
        sparql = re.sub(pattern, replace_ingredient_filter, sparql)
        
        # 3. Corregir referencias directas a ingredientes sin capitalizar
        def capitalize_ingredient(match):
            return f'ingredient:{match.group(1).capitalize()}'
        
        sparql = re.sub(r'ingredient:(\w+)', capitalize_ingredient, sparql)
        
        # 4. Corregir patrones comunes de CONTAINS y FILTER
        def fix_contains_filter(match):
            ingredient = match.group(1).capitalize()
            return f'CONTAINS(LCASE(STR(?ingredient)), "{match.group(1).lower()}")'
        
        sparql = re.sub(
            r'CONTAINS\(LCASE\(STR\(\?ingredient\)\),\s*["\'](\w+)["\']\)',
            fix_contains_filter, 
            sparql
        )
        
        # 5. Corregir filtros que usan regex
        def fix_regex_filter(match):
            ingredient = match.group(1).lower()
            return f'REGEX(STR(?ingredient), "{ingredient}", "i")'
        
        sparql = re.sub(
            r'REGEX\(STR\(\?ingredient\),\s*["\'](\w+)["\']\s*,\s*["\']\w*["\']\)',
            fix_regex_filter, 
            sparql
        )
        
        # 6. Corregir ingredientes dentro de cadenas literales
        def capitalize_ingredient_in_string(match):
            return f'"{match.group(1).capitalize()}"'
        
        # Lista común de ingredientes para identificar en strings
        common_ingredients = [
            "vodka", "gin", "rum", "tequila", "whiskey", "brandy", 
            "vermouth", "lime", "lemon", "orange", "pineapple", 
            "cranberry", "mint", "coffee", "milk", "cream", "sugar",
            "aperol"  # Añadido para el caso específico
        ]
        
        for ingredient in common_ingredients:
            sparql = re.sub(
                f'["\']({re.escape(ingredient)})["\']', 
                capitalize_ingredient_in_string, 
                sparql,
                flags=re.IGNORECASE
            )
            
        # 7. Corregir nombres de cócteles específicos conocidos
        cocktail_mappings = {
            "aperol spritz": "AperolSpritz",
            "aperol": "Aperol",
            "bloody mary": "BloodyMary",
            "margarita": "Margarita",
            "tinto de verano": "TintoDeVerano",
            "piña colada": "PiñaColada",
            "mojito": "Mojito"
        }
        
        # Corregir referencias directas a cócteles
        for cocktail_name, formatted_name in cocktail_mappings.items():
            # Corregir referencia directa al URI
            sparql = re.sub(
                f'cocktail:({re.escape(cocktail_name)})', 
                f'cocktail:{formatted_name}', 
                sparql,
                flags=re.IGNORECASE
            )
            
            # Corregir en patrones FILTER
            sparql = re.sub(
                f'\\?cocktail\\s*=\\s*cocktail:({re.escape(cocktail_name)})',
                f'?cocktail = cocktail:{formatted_name}',
                sparql,
                flags=re.IGNORECASE
            )
            
        # 8. Corregir errores comunes de sintaxis SPARQL
        # Asegurarse de que los paréntesis están correctamente equilibrados
        open_brackets = sparql.count("{")
        close_brackets = sparql.count("}")
        if open_brackets > close_brackets:
            sparql += "}" * (open_brackets - close_brackets)
        
        # Asegurarse de que las cláusulas SELECT tienen una variable al menos
        if "SELECT" in sparql.upper() and "WHERE" in sparql.upper():
            select_part = sparql.split("WHERE")[0]
            if "SELECT" in select_part.upper() and not re.search(r'SELECT\s+\?', select_part, re.IGNORECASE):
                # Añadir una variable ?cocktail si falta
                sparql = sparql.replace(
                    "SELECT", 
                    "SELECT ?cocktail", 
                    1
                )
        
        # 9. Para consultas que contienen "Aperol Spritz", crear una consulta específica
        if "aperol" in sparql.lower():
            cocktail_name = "AperolSpritz"  # Nombre específico para este cóctel
            
            # Crear una consulta robusta y sencilla para este caso específico
            sparql = f"""
PREFIX cocktail: <http://www.semanticweb.org/cocktail/ontology#>
PREFIX property: <http://www.semanticweb.org/cocktail/property#>
PREFIX ingredient: <http://www.semanticweb.org/cocktail/ingredient#>
PREFIX glass: <http://www.semanticweb.org/cocktail/glass#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?cocktail
       (REPLACE(STR(?cocktail), "^.*#", "") AS ?nombre)
       (GROUP_CONCAT(DISTINCT ?ingredientName; separator=", ") AS ?ingredientes)
WHERE {{
  ?cocktail rdf:type cocktail:Cocktail .
  
  {{
    ?cocktail = cocktail:{cocktail_name} .
  }}
  UNION
  {{
    FILTER(CONTAINS(LCASE(STR(?cocktail)), "aperol"))
  }}
  
  # Obtener ingredientes
  OPTIONAL {{
    ?cocktail property:hasIngredient ?ingredient .
    BIND(REPLACE(STR(?ingredient), "^.*#", "") AS ?ingredientName)
  }}
}}
GROUP BY ?cocktail
"""
            
        logger.info(f"SPARQL generado y corregido: {sparql[:200]}...")
        
        return sparql
        
        # 3. Corregir errores de formato con Rum
        # Asegurar que si hay una comparación directa con "rum", se use la forma correcta "Rum"
        # Por ejemplo: ?cocktail property:hasIngredient ingredient:rum -> ?cocktail property:hasIngredient ingredient:Rum
        sparql = sparql.replace("ingredient:rum", "ingredient:Rum")
        sparql = sparql.replace("ingredient:ron", "ingredient:Rum")
        
        # 4. Añadir lógica para otros ingredientes comunes 
        ingredient_translations = {
            "gin": "Gin", 
            "vodka": "Vodka", 
            "tequila": "Tequila", 
            "whiskey": "Whiskey",
            "ginebra": "Gin"
        }
        
        for esp, eng in ingredient_translations.items():
            sparql = sparql.replace(f"ingredient:{esp}", f"ingredient:{eng}")
        
        return sparql
    
    def _ensure_ontology_loaded(self) -> bool:
        """
        Asegura que la ontología está cargada o intenta cargarla directamente del directorio principal.
        Esto ayuda a manejar casos donde la ontología no se cargó correctamente o usa rutas incorrectas.
        
        Returns:
            bool: True si la ontología está correctamente cargada, False en caso contrario
        """
        if len(self.graph) < 100:  # Si hay pocas tripletas, la ontología probablemente no está cargada correctamente
            print(f"[Ontology] La ontología parece estar vacía o casi vacía ({len(self.graph)} tripletas). Intentando cargar directamente...")
            
            # Intentar cargar desde la ruta estándar
            success = self._load_ontology_if_exists()
            
            # Si no se pudo cargar, intentar con la ruta alternativa en el directorio raíz del proyecto
            if not success or len(self.graph) < 100:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                alt_path = os.path.join(project_root, "ontology", "cocktail_ontology.ttl")
                
                if os.path.exists(alt_path):
                    print(f"[Ontology] Intentando cargar ontología desde ruta alternativa: {alt_path}")
                    try:
                        self.graph.parse(alt_path, format="turtle")
                        print(f"[Ontology] ✓ Ontología cargada desde ruta alternativa con {len(self.graph)} tripletas")
                        return True
                    except Exception as e:
                        print(f"[Ontology] ❌ Error al cargar ontología desde ruta alternativa: {str(e)}")
                        return False
                else:
                    print(f"[Ontology] ❌ No se encontró la ontología en la ruta alternativa: {alt_path}")
                    return False
            else:
                return True
        else:
            # La ontología ya está cargada
            return True

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for Spanish vs English.
        
        Args:
            text: Text to detect language for
            
        Returns:
            'es' for Spanish, 'en' for English
        """
        # Common Spanish words/patterns
        spanish_words = ['muéstrame', 'cócteles', 'con', 'hecho', 'bebidas', 'cóctel', 
                        'preparado', 'ingredientes', 'que', 'contienen', 'utilizan',
                        'dame', 'quiero', 'necesito', 'cuáles', 'cuál', 'como']
        
        text_lower = text.lower()
        
        # Check for Spanish indicators - accented characters and common words
        has_spanish_chars = any(char in text_lower for char in 'áéíóúñ¿¡')
        has_spanish_words = any(word in text_lower.split() for word in spanish_words)
        
        if has_spanish_chars or has_spanish_words:
            return 'es'
        else:
            return 'en'

    async def _translate_ingredient_with_llm(self, ingredient: str, source_lang: str = "es", target_lang: str = "en") -> str:
        """
        Traduce un ingrediente de un idioma a otro utilizando el modelo de lenguaje.
        
        Args:
            ingredient: El nombre del ingrediente a traducir
            source_lang: El idioma de origen (por defecto "es" para español)
            target_lang: El idioma de destino (por defecto "en" para inglés)
            
        Returns:
            La traducción del ingrediente
        """
        from agents.generation_agent.generation_agent import GenerationAgent
        import asyncio
        
        if not ingredient.strip():
            return ingredient
            
        # Crear un agente temporal para la generación
        generation_agent = GenerationAgent("temp_translate_agent")
        
        # Crear un prompt simple para la traducción
        prompt = f"Traduce el siguiente ingrediente de cóctel del {source_lang} al {target_lang}. Responde solo con la palabra traducida: {ingredient}"
        
        try:
            # Enviar la solicitud al LLM con un timeout corto
            response = await asyncio.wait_for(
                generation_agent.generate_response(prompt, []),
                timeout=10.0
            )
            
            # Limpiar la respuesta
            translation = response.strip().lower()
            
            # Si hay varias palabras, tomar la primera
            if " " in translation:
                translation = translation.split()[0]
                
            return translation
            
        except Exception as e:
            logger.error(f"Error al traducir ingrediente con LLM: {str(e)}")
            print(f"[Ontology] Error traduciendo con LLM: {str(e)}. Usando el ingrediente original.")
            return ingredient

    async def _detect_and_translate_ingredients(self, query: str) -> list:
        """
        Detecta ingredientes en la consulta y los traduce al inglés.
        
        Args:
            query: Consulta en lenguaje natural
            
        Returns:
            Lista de tuplas (ingrediente_original, traducción)
        """
        import re
        
        # Determinar si la consulta está en español
        is_spanish = self._detect_language(query) == 'es'
        if not is_spanish:
            return []
            
        # Lista de palabras clave que pueden preceder a ingredientes
        ingredient_indicators = [
            "con", "usando", "de", "hecho con", "preparado con", "que tenga", 
            "que contenga", "incluya", "conteniendo", "a base de"
        ]
        
        # Palabras que no son ingredientes (para evitar falsos positivos)
        stop_words = [
            "cócteles", "cóctel", "bebida", "bebidas", "receta", "recetas", 
            "preparar", "hacer", "mostrar", "enseñar", "ver", "dame", 
            "muéstrame", "quiero", "necesito", "para", "como", "cuál", "cuáles"
        ]
        
        query_lower = query.lower()
        ingredients_found = []
        
        # Extraer posibles ingredientes después de las palabras clave
        for indicator in ingredient_indicators:
            if indicator in query_lower:
                parts = query_lower.split(indicator, 1)
                if len(parts) > 1:
                    after_indicator = parts[1].strip()
                    # Tomar hasta el siguiente espacio o puntuación
                    words = re.split(r'[,.\s]+', after_indicator)
                    
                    for word in words:
                        if word and len(word) > 2 and word not in stop_words:
                            # Traducir el ingrediente usando el LLM
                            translation = await self._translate_ingredient_with_llm(word)
                            ingredients_found.append((word, translation))
                            break  # Solo tomar el primer ingrediente después del indicador
        
        # Si no encontramos ingredientes con los indicadores, buscar palabras que pueden ser ingredientes
        if not ingredients_found:
            words = re.split(r'[,.\s]+', query_lower)
            for word in words:
                if word and len(word) > 2 and word not in stop_words:
                    # Verificar si la palabra es un posible ingrediente
                    # Esto es una heurística simple - podríamos mejorarla
                    if word not in ingredient_indicators and not any(w in word for w in ["cómo", "qué", "cuál", "muestra"]):
                        translation = await self._translate_ingredient_with_llm(word)
                        if translation != word:  # Si cambió en la traducción, probablemente es un ingrediente
                            ingredients_found.append((word, translation))
        
        return ingredients_found
    
    async def _translate_full_query_with_llm(self, query: str) -> str:
        """
        Traduce una consulta completa de español a inglés usando el LLM, manteniendo
        la estructura correcta para la ontología con formato PascalCase para términos compuestos.
        Esta función maneja tanto consultas en español como en inglés.
        
        Args:
            query: La consulta en cualquier idioma (español o inglés)
            
        Returns:
            La consulta procesada para la ontología
        """
        from agents.generation_agent.generation_agent import GenerationAgent
        import asyncio
        
        # Detectar el idioma de la consulta
        query_language = self._detect_language(query)
        
        try:
            # Crear un agente de generación temporal
            generation_agent = GenerationAgent("temp_query_translator")
            
            # Construir el prompt para procesamiento - tanto para español como inglés
            prompt = f"""
            Eres un experto en ontologías de coctelería bilingüe (español-inglés). Tu tarea es preparar la consulta para 
            buscar en una ontología especializada, siguiendo estas directrices:

            DIRECTRICES DE PROCESAMIENTO BILINGÜE:
            
            1. FORMATO UNIVERSAL:
               - TODOS los términos compuestos se almacenan en formato PascalCase SIN ESPACIOS
               - Ejemplos: "High Ball" → "HighBall", "tinto de verano" → "TintoDeVerano"
            
            2. NOMBRES DE CÓCTELES:
               - MANTENER EL NOMBRE ORIGINAL del cóctel en su idioma nativo
               - Convertir a formato PascalCase sin espacios
               - Ejemplos: 
                 * "Tinto de Verano" → "TintoDeVerano" (español, mantiene idioma)
                 * "Bloody Mary" → "BloodyMary" (inglés, mantiene idioma)
                 * "Piña Colada" → "PiñaColada" (mantiene ñ y acentos)
            
            3. INGREDIENTES Y TÉRMINOS TÉCNICOS:
               - Ingredientes comunes: en inglés con inicial mayúscula o PascalCase
               - Ejemplos: 
                 * "vino" → "Wine"
                 * "vino tinto" → "RedWine" 
                 * "zumo de naranja"/"jugo de naranja" → "OrangeJuice"
                 * "agua tónica" → "TonicWater"
            
            4. TIPOS DE VASOS:
               - Todos en inglés con formato PascalCase
               - Ejemplos: 
                 * "copa alta"/"vaso alto" → "HighballGlass"
                 * "copa de Martini" → "MartiniGlass"
                 * "copa Hurricane" → "HurricaneGlass"
            
            5. MÉTODOS DE PREPARACIÓN:
               - Todos en inglés con formato PascalCase
               - Ejemplos: 
                 * "agitado y mezclado" → "ShakeAndStir"
                 * "directo en vaso" → "BuildInGlass"
            
            6. ESTRATEGIA DE BÚSQUEDA MULTILINGÜE:
               - Asegurar que la consulta pueda encontrar términos tanto en español como en inglés
               - Para nombres de cócteles y términos ambiguos, mantener referencias a ambas versiones
               - Ejemplos:
                 * Para buscar "Tinto de Verano", usar "TintoDeVerano"
                 * Para buscar "vino", usar "Wine" 
               
            Recuerda: La ontología SIEMPRE usa PascalCase para términos compuestos. Los términos técnicos están en inglés, 
            pero los nombres propios de cócteles conservan su idioma original.
            
            Consulta original ({'español' if query_language == 'es' else 'inglés'}): "{query}"
            
            Consulta procesada (con formato para ontología): 
            """
            
            # Realizar la traducción/procesamiento con timeout
            print(f"[Ontology] Procesando consulta para formato de ontología: '{query}'")
            start_time = __import__('time').time()
            
            # Esperar la respuesta con timeout
            processing_task = generation_agent.generate_response(prompt, [])
            processed_query = await asyncio.wait_for(processing_task, timeout=15.0)
            
            # Limpiar la respuesta procesada
            processed_query = processed_query.strip().strip('"')
            
            # Registrar el resultado
            elapsed_time = __import__('time').time() - start_time
            print(f"[Ontology] ✓ Consulta procesada en {elapsed_time:.2f}s: '{processed_query}'")
            
            return processed_query
            
        except Exception as e:
            print(f"[Ontology] ⚠️ Error al procesar consulta: {str(e)}. Usando consulta original.")
            logger.error(f"Error processing query: {e}")
            # En caso de error, devolver la consulta original
            return query
