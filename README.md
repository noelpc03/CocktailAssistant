
# Sistema de Recuperación de Información Multiagente para Bartenders

Un sistema avanzado de recuperación de información y generación de respuestas basado en una arquitectura multiagente.

## Descripción

Este proyecto implementa un sistema de recuperación de información especializado en coctelería y bartenders. Utiliza una arquitectura multiagente donde cada componente del sistema funciona como un agente autónomo que coopera con los demás a través de mensajes.

## Arquitectura Multiagente

El sistema está compuesto por los siguientes agentes:

### 1. Agente Coordinador (Coordinator Agent)

-**Función**: Orquestar la comunicación entre todos los agentes

-**Responsabilidades**: Inicializar el sistema, dirigir las solicitudes a los agentes apropiados, coordinar el flujo de trabajo

### 2. Agente de Recolección (Crawler Agent)

-**Función**: Extraer información de páginas web

-**Responsabilidades**: Realizar crawling BFS (Breadth-First Search) de páginas web sobre cócteles, partiendo de URLs semilla, obtener contenido relevante, limpiar y estructurar la información

-**Características del Crawler BFS**:
  - Utiliza algoritmo de búsqueda en amplitud (BFS) para explorar páginas web
  - Comienza desde URLs semilla específicas sobre cócteles
  - Control de profundidad máxima para limitar la exploración
  - Extracción de enlaces y normalización de URLs
  - Prevención de bucles mediante registro de URLs ya visitadas

### 3. Agente de Vectorización (Vectorizer Agent)

-**Función**: Convertir documentos de texto a embeddings vectoriales

-**Responsabilidades**: Dividir documentos en fragmentos, generar embeddings utilizando modelos de transformers

### 4. Agente de Recuperación (Retrieval Agent)

-**Función**: Almacenar y recuperar vectores de documentos

-**Responsabilidades**: Mantener un índice FAISS, realizar búsquedas por similitud de vectores

### 5. Agente de Búsqueda (Search Agent)

-**Función**: Procesar consultas y formatear resultados

-**Responsabilidades**: Vectorizar consultas, formatear resultados para presentación al usuario

### 6. Agente de Generación (Generation Agent)

-**Función**: Generar respuestas utilizando modelos de lenguaje

-**Responsabilidades**: Integrar con Google Gemini, construir prompts con el contexto relevante

## Requisitos

- Python 3.9+
- Bibliotecas requeridas en `requirements.txt`

## Instalación

1. Clonar el repositorio:

```bash

gitclone [URL_DEL_REPOSITORIO]

cdia-sri-sim

```

2. Instalar dependencias:

```bash

pipinstall-rrequirements.txt

```

3. Configurar la API key para Gemini:

- Crear un archivo `tokenGemini.txt` en la raíz del proyecto con la API key de Gemini

## Uso

### Ejecutar el sistema completo

```bash

./agent_system.shsearch"¿Cómo preparar un Martini?"

```

### Ejecutar solo la búsqueda sin generación LLM

```bash

./agent_system.shsearch"Ingredientes del Manhattan"--no-llm

```

### Iniciar el proceso de crawling y indexación

```bash

./agent_system.shcrawl--urls"https://www.url1.com""https://www.url2.com"

```

## Configuración

La configuración del sistema se encuentra en `agents/config.json`. Aquí se pueden modificar:

- URLs predeterminadas para crawling
- Modelo de embeddings utilizado
- Número máximo de resultados
- Parámetros del modelo de generación
- Otros parámetros de cada agente

## Estructura del Proyecto

```

agents/

├── common/              # Componentes compartidos entre agentes

│   ├── agent_interface.py   # Interfaz base para todos los agentes

│   ├── message_broker.py    # Sistema de mensajería

│   ├── data_store.py        # Almacén de datos compartido

│   └── config_manager.py    # Gestor de configuración

├── coordinator_agent/   # Agente coordinador

├── crawler_agent/       # Agente de recolección

├── vectorizer_agent/    # Agente de vectorización

├── retrieval_agent/     # Agente de recuperación

├── search_agent/        # Agente de búsqueda

├── generation_agent/    # Agente de generación

└── data/                # Datos generados por los agentes

    └── embeddings/      # Embeddings vectoriales almacenados

```

## Comunicación entre Agentes

Los agentes se comunican a través de un broker de mensajes centralizado. Cada agente puede enviar y recibir mensajes, y el broker se encarga de entregar los mensajes a los destinatarios correctos. La comunicación es asíncrona, lo que permite operaciones paralelas y mayor eficiencia.

## Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

-**Nuevos Agentes**: Se pueden añadir nuevos agentes implementando la interfaz `Agent`

-**Nuevos Modelos**: Se pueden integrar diferentes modelos de embeddings o LLMs

-**Nuevas Fuentes**: Se pueden añadir más fuentes de información al crawler
