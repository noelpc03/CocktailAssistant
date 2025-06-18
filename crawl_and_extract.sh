#!/bin/bash

# Script para ejecutar el crawler seguido por la extracción de ontología
set -e  # Detener el script si algún comando falla

# Obtener el directorio de este script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================================================="
echo "PASO 1: Ejecutando el crawler para recolectar información de cócteles"
echo "==================================================================="

# Ejecutar el crawler
bash "$SCRIPT_DIR/agent_system.sh" crawl

echo ""
echo "==================================================================="
echo "PASO 2: Ejecutando la extracción de triples de la ontología"
echo "==================================================================="

# Ejecutar la extracción de ontología
bash "$SCRIPT_DIR/agent_system.sh" ontology extract

echo ""
echo "==================================================================="
echo "¡Proceso completado! Datos recolectados y tripletas de ontología extraídos."
echo "==================================================================="
