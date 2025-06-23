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
        
        # Procesar cada propiedad de datos


    # El método process_message se ha movido al archivo process_message.py
        
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
        
    def generate_ontology_schema(self) -> Dict[str, Any]:
        """
        Analiza la estructura de la ontología y genera un esquema utilizando métodos nativos de RDFlib.
        El esquema incluye clases, propiedades de objeto, propiedades de datos y relaciones entre clases.
        
        Returns:
            Diccionario con el esquema de la ontología
        """
        import datetime
        
        schema = {
            "classes": [],
            "objectProperties": [],
            "dataProperties": [],
            "classRelations": [],
            "statistics": {
                "totalTriples": len(self.graph),
                "generatedAt": datetime.datetime.now().isoformat()
            }
        }
        
        # Extraer clases
        classes = set()
        for subject in self.graph.subjects(RDF.type, OWL.Class):
            classes.add(subject)
        
        for subject in self.graph.subjects(RDF.type, RDFS.Class):
            classes.add(subject)
        
        # También buscar clases que no estén explícitamente declaradas
        for _, _, obj in self.graph.triples((None, RDF.type, None)):
            if obj != OWL.Class and obj != RDFS.Class and obj != OWL.ObjectProperty and obj != OWL.DatatypeProperty:
                classes.add(obj)
        
        # Procesar cada clase
        for cls in classes:
            class_uri = str(cls)
            class_name = class_uri.split('#')[-1] if '#' in class_uri else class_uri.split('/')[-1]
            
            # Obtener etiqueta y comentarios
            label = None
            for lbl in self.graph.objects(cls, RDFS.label):
                label = str(lbl)
                break
                
            comment = None
            for cmt in self.graph.objects(cls, RDFS.comment):
                comment = str(cmt)
                break
                
            # Obtener superclases
            superclasses = []
            for parent in self.graph.objects(cls, RDFS.subClassOf):
                parent_uri = str(parent)
                parent_name = parent_uri.split('#')[-1] if '#' in parent_uri else parent_uri.split('/')[-1]
                superclasses.append(parent_name)
                
            # Obtener subclases
            subclasses = []
            for child in self.graph.subjects(RDFS.subClassOf, cls):
                child_uri = str(child)
                child_name = child_uri.split('#')[-1] if '#' in child_uri else child_uri.split('/')[-1]
                subclasses.append(child_name)
                
            schema["classes"].append({
                "name": class_name,
                "uri": class_uri,
                "label": label or class_name,
                "description": comment or "",
                "superClasses": superclasses,
                "subClasses": subclasses
            })
        
        # Extraer propiedades de objeto
        for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            prop_uri = str(prop)
            prop_name = prop_uri.split('#')[-1] if '#' in prop_uri else prop_uri.split('/')[-1]
            
            # Obtener etiqueta y comentarios
            label = None
            for lbl in self.graph.objects(prop, RDFS.label):
                label = str(lbl)
                break
                
            comment = None
            for cmt in self.graph.objects(prop, RDFS.comment):
                comment = str(cmt)
                break
                
            # Obtener dominio y rango
            domains = []
            for domain in self.graph.objects(prop, RDFS.domain):
                domain_name = domain.split('#')[-1] if '#' in domain else domain.split('/')[-1]
                domains.append(domain_name)
                
            ranges = []
            for range_val in self.graph.objects(prop, RDFS.range):
                range_name = range_val.split('#')[-1] if '#' in range_val else range_val.split('/')[-1]
                ranges.append(range_name)
                
            schema["objectProperties"].append({
                "name": prop_name,
                "uri": prop_uri,
                "label": label or prop_name,
                "description": comment or "",
                "domains": domains,
                "ranges": ranges
            })
        
        # Extraer propiedades de datos
        for prop in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            prop_uri = str(prop)
            prop_name = prop_uri.split('#')[-1] if '#' in prop_uri else prop_uri.split('/')[-1]
            
            # Obtener etiqueta y comentarios
            label = None
            for lbl in self.graph.objects(prop, RDFS.label):
                label = str(lbl)
                break
                
            comment = None
            for cmt in self.graph.objects(prop, RDFS.comment):
                comment = str(cmt)
                break
                
            # Obtener dominio y rango
            domains = []
            for domain in self.graph.objects(prop, RDFS.domain):
                domain_name = domain.split('#')[-1] if '#' in domain else domain.split('/')[-1]
                domains.append(domain_name)
                
            ranges = []
            for range_val in self.graph.objects(prop, RDFS.range):
                range_name = range_val.split('#')[-1] if '#' in range_val else range_val.split('/')[-1]
                ranges.append(range_name)
                
            schema["dataProperties"].append({
                "name": prop_name,
                "uri": prop_uri,
                "label": label or prop_name,
                "description": comment or "",
                "domains": domains,
                "ranges": ranges
            })
        
        # Extraer relaciones entre clases a partir de las propiedades de objeto
        for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            domains = list(self.graph.objects(prop, RDFS.domain))
            ranges = list(self.graph.objects(prop, RDFS.range))
            
            if domains and ranges:
                for domain in domains:
                    for range_val in ranges:
                        schema["classRelations"].append({
                            "from": domain.split('#')[-1],
                            "to": range_val.split('#')[-1],
                            "property": str(prop).split('#')[-1]
                        })
        
        # Añadir algunas estadísticas adicionales
        schema["statistics"].update({
            "classCount": len(schema["classes"]),
            "objectPropertyCount": len(schema["objectProperties"]),
            "dataPropertyCount": len(schema["dataProperties"]),
            "relationCount": len(schema["classRelations"])
        })
        
        # Guardar el esquema en un archivo
        schema_file = os.path.join(self.ontology_dir, "ontology_schema.json")
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated ontology schema with {len(schema['classes'])} classes, " +
                   f"{len(schema['objectProperties'])} object properties, " +
                   f"{len(schema['dataProperties'])} data properties")
        
        return schema

    # Implementación de procesamiento de consultas en lenguaje natural
    async def process_query(self, query_text: str, language: str = None) -> Dict[str, Any]:
        """
        Process a natural language query against the ontology.
        
        Args:
            query_text: The natural language query
            language: Optional language code ('es' or 'en'), will be auto-detected if not provided
            
        Returns:
            Dictionary with query results and natural language answer
        """
        from agents.ontology_agent.ontology_query import OntologyQuerySystem
        from agents.common.mistral_client import MistralClient
        
        # Ensure ontology is loaded
        if not self._ensure_ontology_loaded():
            return {
                "status": "error",
                "message": "Failed to load ontology",
                "answer": "No se pudo cargar la ontología para responder a tu pregunta."
            }
        
        try:
            # Initialize LLM client if needed
            llm_client = None
            try:
                llm_client = MistralClient()
                logger.info("Initialized Mistral client for ontology query processing")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral client, using basic query processing: {e}")
            
            # Initialize the query system
            query_system = OntologyQuerySystem(ontology_agent=self, llm_client=llm_client)
            
            # Process the query
            response = await query_system.query(query_text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing ontology query: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "answer": "Lo siento, ocurrió un error al procesar tu consulta."
            }
    
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
            
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        Import the actual implementation from process_message module.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        # Import here to avoid circular imports
        from agents.ontology_agent.process_message import process_message as process_message_impl
        return await process_message_impl(self, message)
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
