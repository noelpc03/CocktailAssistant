#!/bin/bash
# Script para iniciar la aplicación web de Cocktail Assistant

echo "Iniciando Cocktail Assistant..."
echo "Instalando dependencias necesarias..."

# Asegurarse de que las dependencias estén instaladas
pip install -q -r requirements.txt

echo "Iniciando la aplicación web..."
streamlit run app.py

# En caso de error
if [ $? -ne 0 ]; then
    echo "Error al iniciar la aplicación. Asegúrate de tener streamlit instalado:"
    echo "pip install streamlit"
fi
