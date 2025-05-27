#!/bin/bash
# filepath: /home/noel/Disco D/3er Anno/2do semestre/SRI/ia-sri-sim/search.sh

# Script para ejecutar el módulo de recuperación de información sobre bartenders

# Obtener la ruta base del proyecto (directorio donde está este script)
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"

# Activar entorno virtual si existe
if [ -d "${PROJECT_DIR}/.venv" ]; then
    echo "Activando entorno virtual..."
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# Ejecución del módulo de recuperación
echo "Iniciando el proceso de recuperación de información..."
python -m src.retrieval.main

# Comprobar si la ejecución fue exitosa
if [ $? -eq 0 ]; then
    echo ""
    echo "Proceso de recuperación completado correctamente."
    echo "Los datos han sido vectorizados y almacenados en el índice FAISS."
    echo ""
    echo "Para realizar búsquedas, utilice:"
    echo "python -m src.search.search_app \"su consulta aquí\""
else
    echo ""
    echo "Error durante el proceso de recuperación."
    echo "Revise los logs en bartender_retrieval.log para más detalles."
fi

# Desactivar entorno virtual si fue activado
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi