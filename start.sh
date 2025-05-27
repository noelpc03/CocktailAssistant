#!/bin/bash

# Script para ejecutar la búsqueda de información sobre bartenders

# Ruta al directorio base del proyecto
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Activar entorno virtual si existe
if [ -d "${PROJECT_DIR}/venv" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
    echo "Entorno virtual activado."
fi

# Ir al directorio del proyecto
cd "${PROJECT_DIR}"

# Ejecutar la aplicación de búsqueda
cd "${PROJECT_DIR}"
PYTHONPATH="${PROJECT_DIR}" python3 "${PROJECT_DIR}/src/search/search_app.py" "$@"

# Imprimir un mensaje para el usuario
if [ $? -eq 0 ]; then
    echo ""
    echo "Búsqueda completada usando Gemini 1.5 Flash para la generación de respuestas."
    echo "Para realizar otra búsqueda ejecute ./search.sh con su consulta o use --no-llm para desactivar el LLM."
    echo "Ejemplo: ./search.sh \"cómo hacer un mojito\""
fi