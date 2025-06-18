
# Cocktail Assistant: Sistema de Recuperación de Información Multiagente para Bartenders

## Autores
- Amalia Beatriz Valiente Hinojosa C312
- Jorge Alejandro Echevarr´ıa Brunet C312
- Rodrigo Mederos Gonz´alez C311 
- Noel P´erez Calvo C311

## Descripción

El proyecto implementa un sistema de recuperación de información especializado en el dominio de bartender, implementado mediante una arquitectura multiagente alineada con el modelo Retrieve-Augmented Generation (RAG). El sistema integra diversos componentes que incluyen: procesamiento de lenguaje natural para la interpretación de consultas y generación de respuestas, técnicas metaheurísticas para optimizar la recuperación de información, representación del conocimiento mediante una ontología especializada, y un crawler automatizado para la recopilación y actualización dinámica de datos.
 
## Requerimientos
- Python 3.9
- Conexión a Internet para el proceso de crawling y uso de APIs externas
- Instalación de las bibliotecas de requirements.txt
- Token de Mistral

## API's
- Mistral
- DuckDuckGo-Search 

## Instalacion de requisitos
```bash
pip install -r requirements.txt
```
## Recuperación de información
```bash
./crawl_and_extract.sh
```
## Iniciar el proyecto con interfaz web
```bash
./startup.sh
```