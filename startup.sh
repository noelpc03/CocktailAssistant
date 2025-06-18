#!/bin/bash
# Script para iniciar la aplicación web de Cocktail Assistant

echo "Iniciando Cocktail Assistant..."
echo "Instalando dependencias necesarias..."

# Asegurarse de que las dependencias estén instaladas
pip install -q -r requirements.txt

echo "Verificando que el sistema esté listo para usar el agente de decisión..."
# Verificar si la ontología está generada
if [ ! -f "./ontology/cocktail_ontology.owl" ]; then
    echo "ATENCIÓN: No se ha detectado la ontología. Se recomienda ejecutar primero:"
    echo "./agent_system.sh extract_ontology"
    echo ""
    echo "Continuando de todos modos..."
fi

echo "Iniciando la aplicación web..."
# Asegurarse de que estamos en el directorio correcto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${SCRIPT_DIR}/app/app.py"

# Verificar si el archivo app.py existe en la ruta especificada
if [ -f "$APP_PATH" ]; then
    echo "Ejecutando aplicación desde: $APP_PATH"
    # Establecemos el directorio de trabajo como la raíz del proyecto
    cd "$SCRIPT_DIR"
    streamlit run "$APP_PATH"
else
    echo "ERROR: No se encuentra el archivo app.py en $APP_PATH"
    exit 1
fi

# En caso de error
if [ $? -ne 0 ]; then
    echo "Error al iniciar la aplicación. Asegúrate de tener streamlit instalado:"
    echo "pip install streamlit"
fi
