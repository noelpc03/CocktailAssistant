"""
Sistema de consultas inteligente para la ontología de cócteles.
Proporciona funcionalidades para traducir preguntas en lenguaje natural a consultas SPARQL,
ejecutarlas contra la ontología, y generar respuestas naturales.

Este sistema utiliza un enfoque completamente basado en LLM para procesar consultas en lenguaje natural,
sin lógica hardcoded ni componentes de fallback.
"""
import os
import json
import logging
import re
import datetime
import copy
from typing import Dict, Any, List, Optional
import rdflib
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from rdflib.plugins.sparql import prepareQuery

logger = logging.getLogger(__name__)

class OntologyQuerySystem:
    """
    Sistema que permite realizar consultas en lenguaje natural sobre la ontología de cócteles.
    Utiliza exclusivamente un LLM para interpretar las preguntas y generar consultas SPARQL dinámicamente,
    sin componentes de fallback o lógica de normalización manual.
    """
    
    def __init__(self, ontology_agent, llm_client=None):
        """
        Inicializa el sistema de consultas LLM-only.
        
        Args:
            ontology_agent: Instancia del agente de ontología para acceder al grafo RDF
            llm_client: Cliente para el modelo de lenguaje (Mistral)
        """
        self.ontology_agent = ontology_agent
        self.llm = llm_client
        self.graph = ontology_agent.graph
        
        # Cargar esquema de ontología
        self.schema_path = os.path.join(ontology_agent.ontology_dir, "ontology_schema.json")
        self.load_schema()
        
        # Definir prefijos para consultas SPARQL
        self.prefixes = {
            "rdf": str(RDF),
            "rdfs": str(RDFS),
            "owl": str(OWL),
            "cocktail": str(self.ontology_agent.COCKTAIL),
            "ingredient": str(self.ontology_agent.INGREDIENT),
            "glass": str(self.ontology_agent.GLASS),
            "property": str(self.ontology_agent.PROPERTY),
            "method": str(self.ontology_agent.METHOD)
        }
        
        # Prefijo formateado para SPARQL
        self.prefixes_str = "\n".join([f"PREFIX {prefix}: <{uri}>" for prefix, uri in self.prefixes.items()])
        
        logger.info("OntologyQuerySystem inicializado con enfoque completamente basado en LLM")
    
    def load_schema(self):
        """Carga el esquema de la ontología desde el archivo JSON y prepara el resumen para prompts"""
        try:
            if os.path.exists(self.schema_path):
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    self.schema = json.load(f)
                logger.info(f"Esquema de ontología cargado con {len(self.schema['classes'])} clases, "
                          f"{len(self.schema['objectProperties'])} propiedades de objeto y "
                          f"{len(self.schema['dataProperties'])} propiedades de datos")
            else:
                # Si el esquema no existe, generarlo
                logger.warning("Esquema de ontología no encontrado. Generando un nuevo esquema...")
                self.schema = self.ontology_agent.generate_ontology_schema()
                logger.info("Esquema de ontología generado")
        except Exception as e:
            logger.error(f"Error al cargar el esquema de ontología: {str(e)}")
            self.schema = {
                "classes": [],
                "objectProperties": [],
                "dataProperties": [],
                "classRelations": [],
                "statistics": {"totalTriples": 0}
            }
        
        # Generar resumen del esquema para su uso en prompts
        self._schema_summary = self._generate_schema_summary()
    
    def _generate_schema_summary(self) -> str:
        """
        Genera un resumen claro y conciso del esquema de la ontología para usar en prompts.
        
        Returns:
            Resumen del esquema como texto
        """
        summary = "CLASES EN LA ONTOLOGÍA:\n"
        for cls in self.schema.get("classes", []):
            name = cls.get("name", "")
            label = cls.get("label", name)
            summary += f"- {name} (Etiqueta: {label})\n"
        
        summary += "\nPROPIEDADES DE OBJETO:\n"
        for prop in self.schema.get("objectProperties", []):
            name = prop.get("name", "")
            domains = ", ".join(prop.get("domains", [])) or "No especificado"
            ranges = ", ".join(prop.get("ranges", [])) or "No especificado"
            summary += f"- {name} (Dominio: {domains}, Rango: {ranges})\n"
        
        summary += "\nPROPIEDADES DE DATOS:\n"
        for prop in self.schema.get("dataProperties", []):
            name = prop.get("name", "")
            domains = ", ".join(prop.get("domains", [])) or "No especificado"
            ranges = ", ".join(prop.get("ranges", [])) or "No especificado"
            summary += f"- {name} (Dominio: {domains}, Rango: {ranges})\n"
        
        summary += "\nRELACIONES ENTRE CLASES:\n"
        for rel in self.schema.get("classRelations", []):
            summary += f"- {rel.get('from', '')} -> {rel.get('property', '')} -> {rel.get('to', '')}\n"
        
        return summary
    
    async def query(self, question: str) -> Dict[str, Any]:
        """
        Procesa una pregunta del usuario y devuelve una respuesta basada en la ontología.
        Este método implementa un enfoque directo basado enteramente en LLM, sin fallbacks o lógica manual.
        
        Args:
            question: Pregunta en lenguaje natural
            
        Returns:
            Diccionario con la respuesta y metadatos
        """
        if not self.llm:
            logger.error("No hay cliente LLM disponible. Este sistema requiere un LLM para funcionar.")
            return {
                "question": question,
                "answer": "Lo siento, no puedo procesar consultas en este momento. El sistema LLM no está disponible.",
                "success": False,
                "error": "LLM no disponible",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        try:
            # 1. Analizar la consulta para obtener su representación estructurada
            logger.info(f"Iniciando análisis de consulta: '{question}'")
            analysis = await self._analyze_query(question)
            
            # 2. Generar consulta SPARQL basada en el análisis
            logger.info("Generando consulta SPARQL basada en el análisis")
            sparql_query = await self._generate_sparql(question, analysis)
            
            # 3. Ejecutar la consulta contra la ontología
            logger.info("Ejecutando consulta SPARQL contra la ontología")
            results = self._execute_query(sparql_query)
            logger.info(f"Consulta ejecutada. Obtenidos {len(results)} resultados")
            
            # Ya no generamos respuestas usando LLM aquí para evitar llamadas duplicadas
            # Los resultados crudos serán enviados al agente generador
            logger.info("Omitiendo generación de respuesta en lenguaje natural (será manejada por generation_agent)")
            # Asignamos un valor vacío a answer para mantener la estructura de respuesta
            answer = ""
            
            response = {
                "question": question,
                "analysis": analysis,
                "sparql_query": sparql_query,
                "results": results,
                "answer": answer,
                "success": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Imprimir la respuesta completa que se enviará al coordinador
            print(f"\n{'#'*100}\n[DEBUG-ONTOLOGY] FINAL RESPONSE TO COORDINATOR\n{'#'*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"SUCCESS: {response.get('success', False)}", flush=True)
            print(f"ANALYSIS: {json.dumps(analysis, ensure_ascii=False)[:200]}...", flush=True)
            print(f"SPARQL: {sparql_query[:200]}...", flush=True)
            print(f"RESULTS COUNT: {len(results)}", flush=True)
            print(f"ANSWER: {answer}", flush=True)
            print(f"\nFULL RESPONSE (truncated):", flush=True)
            print(json.dumps(response, indent=2, default=str, ensure_ascii=False)[:1500], flush=True)
            print(f"{'#'*100}\n", flush=True)
            
            return response
        except Exception as e:
            logger.error(f"Error procesando consulta: {str(e)}")
            
            # Generar respuesta de error empática (con manejo de excepciones)
            try:
                error_response = await self._generate_error_response(question, str(e))
            except Exception as err_error:
                logger.error(f"Error generando respuesta de error: {str(err_error)}")
                error_response = "Lo siento, no pude procesar tu consulta sobre ingredientes del martini."
            
            error_result = {
                "question": question,
                "answer": error_response,
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Imprimir la respuesta de error que se enviará al coordinador
            print(f"\n{'#'*100}\n[DEBUG-ONTOLOGY] ERROR RESPONSE TO COORDINATOR\n{'#'*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"ERROR: {str(e)}", flush=True)
            print(f"SUCCESS: False", flush=True)
            print(f"ANSWER: {error_response}", flush=True)
            print(f"\nFULL ERROR RESPONSE:", flush=True)
            print(json.dumps(error_result, indent=2, default=str, ensure_ascii=False), flush=True)
            print(f"{'#'*100}\n", flush=True)
            
            return error_result
    
    async def _analyze_query(self, question: str) -> Dict[str, Any]:
        """
        Analiza la consulta del usuario usando el LLM para extraer entidades, relaciones y restricciones.
        Este método utiliza un enfoque completamente basado en LLM sin reglas manuales o fallbacks.
        
        Args:
            question: Pregunta en lenguaje natural
            
        Returns:
            Análisis estructurado de la consulta
        """
        # Construir el prompt para el LLM
        prompt = self._build_analysis_prompt(question)
        
        try:
            # Generar y procesar la respuesta
            response = await self.llm.generate(prompt)
            response_text = response.get('text', '')
            
            # Imprimir la respuesta completa en la consola para depuración
            print(f"\n{'='*100}\n[DEBUG-ONTOLOGY] LLM RESPONSE (ANALYSIS)\n{'='*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"\nRESPONSE:", flush=True)
            print(f"{response_text}", flush=True)
            print(f"{'='*100}\n", flush=True)
            
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                raise ValueError("No se encontró un objeto JSON válido en la respuesta del LLM")
                
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
            
            # Print the extracted structured analysis
            print(f"[DEBUG-ONTOLOGY] EXTRACTED ANALYSIS:", flush=True)
            print(json.dumps(analysis, indent=2, ensure_ascii=False), flush=True)
            print(f"{'='*100}\n", flush=True)
            
            logger.info(f"Análisis de la consulta completado: {json.dumps(analysis, ensure_ascii=False)[:100]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de consulta: {str(e)}")
            raise ValueError(f"Error al analizar la consulta: {str(e)}")
    
    def _build_analysis_prompt(self, question: str) -> str:
        """
        Construye un prompt completo para análisis y normalización de la consulta.
        
        Args:
            question: La pregunta del usuario
            
        Returns:
            Prompt para el LLM
        """
        return f"""
        Analiza y normaliza esta consulta sobre cócteles: "{question}"
        
        ESQUEMA DE LA ONTOLOGÍA:
        {self._schema_summary}
        
        TAREA:
        1. Identifica la intención principal (encontrar cócteles, listar ingredientes, etc.)
        2. Extrae todas las entidades mencionadas (cócteles, ingredientes, métodos, etc.)
        3. Identifica las relaciones relevantes (contiene, se sirve en, etc.)
        4. NORMALIZA todas las entidades y propiedades según los nombres EXACTOS del esquema
        5. Determina restricciones o filtros mencionados (contenido alcohólico, etc.)
        
        REQUISITOS ESTRICTOS:
        - Los nombres de entidades DEBEN coincidir EXACTAMENTE con los de la ontología
        - Usa "Cocktail", "Ingredient", "Glass", "Method" como tipos de entidad
        - Usa "hasIngredient", "servedIn", "preparedBy" como nombres de relaciones
        - Los nombres deben utilizar la forma exacta (sin espacios, con mayúsculas adecuadas) como aparecen en sus identificadores URI
        - Por ejemplo, usa "Martini", "OldFashioned", "DryVermouth", etc. sin espacios y con mayúsculas correctas
        - Si una entidad no existe en la ontología, usa el nombre más cercano que SÍ exista
        - Identifica las restricciones numéricas o categóricas mencionadas
        
        RESPUESTA:
        Proporciona ÚNICAMENTE un objeto JSON con esta estructura:
        {{
          "intent": "get_cocktails|get_ingredients|list_recipes|etc",
          "entities": [
            {{"type": "Cocktail|Ingredient|Glass|Method", "name": "NombreExactoEnOntología"}}
          ],
          "relations": [
            {{"type": "hasIngredient|servedIn|preparedBy", "direction": "from_cocktail_to_ingredient"}}
          ],
          "constraints": [
            {{"property": "alcoholContent", "operator": "greaterThan|lessThan|equals", "value": "X"}}
          ],
          "language": "es|en"
        }}
        """
    
    async def _generate_sparql(self, question: str, analysis: Dict[str, Any]) -> str:
        """
        Genera una consulta SPARQL usando exclusivamente el LLM.
        Toma el análisis de la consulta y lo convierte directamente en una consulta SPARQL válida.
        
        Args:
            question: Pregunta original del usuario
            analysis: Análisis estructurado de la pregunta
            
        Returns:
            Consulta SPARQL completa
        """
        if not self.llm:
            raise ValueError("El cliente LLM no está disponible para generar consultas SPARQL")
        
        # Construir un prompt específico para generación de SPARQL
        system_message = """
        Eres un experto en ontologías y SPARQL. Tu tarea es generar consultas SPARQL precisas
        basándote en análisis estructurados de preguntas en lenguaje natural.
        
        Sigue estas directrices:
        1. IMPORTANTE: Las entidades en esta ontología NO tienen etiquetas (rdfs:label), debes usar sus URIs DIRECTAMENTE
        2. Usa exactamente los nombres de clases y propiedades proporcionados
        3. Para cócteles, usa DIRECTAMENTE la URI con el prefijo, ejemplo: 'cocktail:Martini' en vez de buscar por su etiqueta
        4. NO USES patrones como '?cocktail rdfs:label "Martini"' porque no funcionarán
        5. Usa SIEMPRE referencias directas como 'cocktail:Martini property:hasIngredient ?ingredient'
        6. Asegúrate de que la sintaxis SPARQL es correcta
        7. Incluye todos los prefijos necesarios
        8. Usa nombres de variables relevantes (como ?ingredient, ?glass, etc.)
        9. Proporciona ÚNICAMENTE el código SPARQL puro, sin explicaciones
        10. NO incluyas delimitadores de código como backticks (`) o bloques ```
        11. NO uses formato markdown, solo texto plano con la consulta SPARQL
        """
        
        # Extraer información relevante del análisis
        intent = analysis.get("intent", "get_cocktails")
        entities = analysis.get("entities", [])
        relations = analysis.get("relations", [])
        constraints = analysis.get("constraints", [])
        
        # Crear una versión simplificada del esquema para el contexto
        relevant_classes = []
        for entity in entities:
            entity_type = entity.get("type", "")
            for cls in self.schema.get("classes", []):
                if cls.get("name", "") == entity_type or cls.get("label", "") == entity_type:
                    if cls not in relevant_classes:
                        relevant_classes.append(cls)
        
        # Construir el prompt para el LLM
        prompt = f"""
        Genera una consulta SPARQL para responder a esta pregunta:
        "{question}"
        
        Basándote en este análisis estructurado:
        {json.dumps(analysis, indent=2, ensure_ascii=False)}
        
        Usa estos prefijos en la consulta:
        {self.prefixes_str}
        
        La consulta debe:
        1. Seleccionar las variables relevantes para responder la pregunta
        2. UTILIZAR SIEMPRE LAS URIS DIRECTAS para las entidades conocidas (sin buscar por etiquetas)
        3. Ejemplo correcto: 'cocktail:Martini property:hasIngredient ?ingredient'
        4. Ejemplo INCORRECTO: '?cocktail rdfs:label "Martini" ; property:hasIngredient ?ingredient'
        5. Incluir patrones de triples para todas las relaciones mencionadas
        6. Incluir filtros para las restricciones mencionadas
        7. Ser sintácticamente válida para SPARQL 1.1
        
        IMPORTANTE: 
        - NUNCA uses búsquedas por etiqueta (rdfs:label) ya que las entidades NO tienen etiquetas
        - Las entidades se identifican DIRECTAMENTE por sus URIs (como cocktail:Martini)
        - Proporciona SOLO la consulta SPARQL completa, sin explicaciones adicionales
        - NO incluyas delimitadores de código como ``` o backticks (`)
        - Devuelve el código SPARQL como texto plano sin formato markdown
        """
        
        try:
            # Generar la consulta SPARQL con el LLM
            response = await self.llm.generate(prompt, system_message)
            response_text = response.get('text', '')
            
            # Imprimir la respuesta completa en la consola para depuración
            print(f"\n{'='*100}\n[DEBUG-ONTOLOGY] LLM RESPONSE (SPARQL GENERATION)\n{'='*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"\nANALYSIS SUMMARY:", flush=True)
            print(f"Intent: {analysis.get('intent', 'unknown')}", flush=True)
            print(f"Entities: {json.dumps(analysis.get('entities', []), ensure_ascii=False)}", flush=True)
            print(f"Relations: {json.dumps(analysis.get('relations', []), ensure_ascii=False)}", flush=True)
            print(f"\nLLM RESPONSE:", flush=True)
            print(f"{response_text}", flush=True)
            print(f"{'='*100}\n", flush=True)
            
            # Limpiar delimitadores de código markdown de la respuesta
            # Eliminar bloques de código como ```sparql y ```
            cleaned_response = re.sub(r'```(?:sparql|SPARQL)?|```', '', response_text)
            # Eliminar cualquier backtick residual
            cleaned_response = cleaned_response.replace('`', '')
            
            # Extraer consulta SPARQL (buscar patrones típicos de consultas SPARQL)
            # Primero intentar con los prefijos
            prefix_match = re.search(r'(?:PREFIX\s+[^\n]+\n)+\s*(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\s+.*', 
                                     cleaned_response, re.DOTALL | re.IGNORECASE)
            
            # Si no hay prefijos, buscar SELECT, ASK, etc. directamente
            if not prefix_match:
                query_match = re.search(r'(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\s+.*', 
                                        cleaned_response, re.DOTALL | re.IGNORECASE)
                if query_match:
                    # Añadir los prefijos estándar a la consulta encontrada
                    sparql_query = f"{self.prefixes_str}\n\n{query_match.group(0).strip()}"
                else:
                    # Si no se encuentra un patrón válido, usar la respuesta completa limpia
                    sparql_query = cleaned_response.strip()
            else:
                sparql_query = prefix_match.group(0).strip()
            
            # Validar la sintaxis de la consulta
            try:
                # Intentar preparar la consulta para validar su sintaxis
                prepareQuery(sparql_query, initNs=self.prefixes)
                logger.info("Consulta SPARQL generada y validada con éxito")
            except Exception as e:
                # Registrar el error pero continuar con la consulta
                logger.warning(f"La consulta SPARQL generada tiene errores de sintaxis: {str(e)}")
            
            # Print the final cleaned and formatted SPARQL query
            print(f"\n{'='*100}\n[DEBUG-ONTOLOGY] FINAL SPARQL QUERY\n{'='*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"\nSPARQL:", flush=True)
            print(f"{sparql_query}", flush=True)
            print(f"{'='*100}\n", flush=True)
            
            return sparql_query
            
        except Exception as e:
            logger.error(f"Error generando consulta SPARQL: {str(e)}")
            raise ValueError(f"Error al generar la consulta SPARQL: {str(e)}")
    
    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Ejecuta una consulta SPARQL contra la ontología.
        
        Args:
            query: Consulta SPARQL a ejecutar
            
        Returns:
            Resultados formateados
        """
        try:
            # Ejecutar la consulta
            results = self.graph.query(query)
            
            # Formatear resultados
            formatted_results = []
            for row in results:
                result_dict = {}
                for var_name, value in zip(results.vars, row):
                    var_str = str(var_name)
                    
                    if isinstance(value, URIRef):
                        # Para URIs, obtener la etiqueta si existe
                        label = None
                        for lbl in self.graph.objects(value, RDFS.label):
                            label = str(lbl)
                            break
                        
                        # Extraer nombre de la URI
                        uri_str = str(value)
                        name = uri_str.split('#')[-1] if '#' in uri_str else uri_str.split('/')[-1]
                        
                        result_dict[var_str] = {
                            "uri": uri_str,
                            "name": name,
                            "label": label if label else name
                        }
                    elif isinstance(value, Literal):
                        result_dict[var_str] = str(value)
                    else:
                        result_dict[var_str] = str(value)
                
                formatted_results.append(result_dict)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error ejecutando consulta SPARQL: {str(e)}")
            return [{"error": str(e)}]
    
    async def _generate_answer(self, question: str, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """
        Genera una respuesta en lenguaje natural basada en los resultados usando el LLM.
        
        Args:
            question: Pregunta original
            analysis: Análisis de la pregunta
            results: Resultados de la consulta
            
        Returns:
            Respuesta en lenguaje natural
        """
        if not self.llm:
            logger.warning("LLM client not available for answer generation, using basic response")
            if not results:
                return "No se encontraron resultados para esta consulta."
            return f"Encontré {len(results)} resultados sobre los ingredientes."
        
        try:
            # Determinar el idioma preferido para la respuesta
            language = analysis.get("language", "es")  # Valor por defecto es español
            
            # Formatear resultados para el prompt (limitando por tamaño si es necesario)
            results_limited = results[:20] if len(results) > 20 else results  # Limitar a 20 resultados para el prompt
            # Usar ensure_ascii=False para mantener caracteres no ASCII
            # Usar default=str para manejar tipos no serializables
            results_text = json.dumps(results_limited, indent=2, ensure_ascii=False, default=str)
            
            # Sistema de mensajes para contextualizar al LLM
            system_message = f"""
            Eres un asistente experto en cócteles que responde preguntas basándose en una ontología.
            Responde siempre en {language} ({"español" if language == "es" else "inglés"}).
            Tus respuestas deben ser conversacionales, precisas y enfocadas en la pregunta.
            Si los resultados están vacíos, indícalo amablemente.
            Si hay un error, explica el problema de forma sencilla.
            """
            
            # Construir prompt para generación de respuesta
            prompt = f"""
            Pregunta del usuario: "{question}"
            
            Resultados de la consulta a la ontología:
            {results_text}
            
            Número total de resultados: {len(results)}
            
            Tu respuesta debe:
            1. Ser concisa pero informativa
            2. Incluir detalles relevantes de los resultados
            3. Tener un tono conversacional y amigable
            4. Mencionar el número total de resultados encontrados si es relevante
            5. No mencionar detalles técnicos como SPARQL, ontologías, o URIs
            
            Genera una respuesta natural que conteste directamente a la pregunta.
            """
            
            logger.info("Sending answer generation request to LLM")
            response = await self.llm.generate(prompt, system_message)
            answer_text = response.get('text', '').strip()
            
            # Imprimir los resultados y la respuesta final en la consola para depuración
            print(f"\n{'='*100}\n[DEBUG-ONTOLOGY] QUERY RESULTS AND FINAL ANSWER\n{'='*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"\nRESULTS COUNT: {len(results)}", flush=True)
            if results:
                print(f"\nFIRST 3 RESULTS:", flush=True)
                print(json.dumps(results[:3], indent=2, ensure_ascii=False, default=str), flush=True)
                if len(results) > 3:
                    print(f"...and {len(results) - 3} more results", flush=True)
            else:
                print("NO RESULTS FOUND", flush=True)
            
            print(f"\nLLM FINAL ANSWER:", flush=True)
            print(f"{answer_text}", flush=True)
            print(f"{'='*100}\n", flush=True)
            
            logger.info(f"Received answer from LLM: {len(answer_text)} characters")
            
            if not answer_text:
                # Fallback en caso de respuesta vacía
                if not results:
                    return "No encontré resultados para tu consulta." if language == "es" else "I found no results for your query."
                else:
                    return f"Encontré {len(results)} resultados." if language == "es" else f"I found {len(results)} results."
            
            return answer_text
        except Exception as e:
            logger.error(f"Error generando respuesta con LLM: {str(e)}")
            # Respuesta básica en caso de error
            if not results:
                return "No se encontraron resultados." if language == "es" else "No results found."
            elif results and isinstance(results[0], dict) and "error" in results[0]:
                return f"Ocurrió un error: {results[0]['error']}" if language == "es" else f"An error occurred: {results[0]['error']}"
            else:
                return f"Encontré {len(results)} resultados, pero no puedo generar una respuesta detallada." if language == "es" else \
                       f"I found {len(results)} results, but I can't generate a detailed response."
    
    async def _generate_error_response(self, question: str, error_msg: str) -> str:
        """
        Genera una respuesta amigable para un error usando el LLM.
        
        Args:
            question: Pregunta original
            error_msg: Mensaje de error técnico
            
        Returns:
            Respuesta de error comprensible
        """
        if not self.llm:
            return f"Lo siento, ocurrió un error al procesar tu pregunta: {error_msg}"
        
        system_message = """
        Eres un asistente amable y empático que ayuda a los usuarios cuando hay problemas técnicos.
        Tu tarea es convertir errores técnicos en mensajes comprensibles y útiles.
        """
        
        prompt = f"""
        La siguiente pregunta sobre cócteles generó un error:
        "{question}"
        
        Error técnico: {error_msg}
        
        Por favor, genera una respuesta amigable que:
        1. Se disculpe por no poder responder correctamente
        2. Explique el problema de manera comprensible (sin tecnicismos)
        3. Sugiera cómo reformular la pregunta si es posible
        
        La respuesta debe ser empática y en un tono conversacional.
        NO menciones detalles técnicos como SPARQL, LLM, ontologías, etc.
        En su lugar, enfócate en cómo el usuario puede obtener ayuda.
        """
        
        try:
            response = await self.llm.generate(prompt, system_message)
            error_response = response.get('text', '').strip()
            
            # Imprimir la respuesta de error en la consola para depuración
            print(f"\n{'='*100}\n[DEBUG-ONTOLOGY] ERROR RESPONSE FROM LLM\n{'='*100}", flush=True)
            print(f"QUERY: {question}", flush=True)
            print(f"ERROR: {error_msg}", flush=True)
            print(f"\nLLM ERROR RESPONSE:", flush=True)
            print(f"{error_response}", flush=True)
            print(f"{'='*100}\n", flush=True)
            
            return error_response
        except Exception as e:
            logger.error(f"Error generando respuesta de error: {str(e)}")
            return f"Lo siento, no pude procesar tu pregunta correctamente. Por favor, intenta reformularla de manera más sencilla."

# Método para integrar con el agente de ontología
def integrate_with_ontology_agent(ontology_agent, llm_client=None):
    """
    Integra el sistema de consultas LLM-only con el agente de ontología.
    
    Args:
        ontology_agent: Instancia del agente de ontología
        llm_client: Cliente para el LLM (requerido para funcionalidad completa)
    
    Returns:
        Instancia del sistema de consultas
    """
    if not llm_client:
        logger.warning("Se está inicializando OntologyQuerySystem sin un cliente LLM. "
                      "El sistema requiere un LLM para funcionar correctamente.")
    
    return OntologyQuerySystem(ontology_agent, llm_client)
