import streamlit as st
import subprocess
import json
import os
import time

# Configuración de la página
st.set_page_config(
    page_title="Cocktail Assistant",
    page_icon="🍹",
    layout="wide",
)

# Estilo personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 15px;
    }
    .search-container {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        min-height: 200px;
    }
    .cocktail-header {
        color: #1E3A8A;
        font-family: 'Georgia', serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-weight: 500;
    }
    .st-emotion-cache-6qob1r {
        padding-top: 3rem;
    }
    .answer-text {
        font-size: 17px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Función para ejecutar comandos del sistema
def run_command(command):
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Error al procesar tu consulta")
        print(f"Error de comando: {e}")
        print(f"Error stderr: {e.stderr}")
        return None

# Inicializar estado de sesión
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Encabezado con logo y título
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🍹")
with col2:
    st.markdown('<h1 class="cocktail-header">Cocktail Assistant</h1>', unsafe_allow_html=True)

st.markdown('<p style="font-size: 18px;">Tu asistente personal de cocteles. Pregunta cualquier cosa sobre recetas, ingredientes y más.</p>', unsafe_allow_html=True)
st.markdown("---")

# Contenedor para la interfaz de búsqueda
with st.container():
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Opciones de búsqueda
    search_type = st.radio(
        "Método de búsqueda:",
        ["Simple", "Semántica", "Inteligente (Ontología)"],
        horizontal=True,
        help="Simple: búsqueda por palabras clave. Semántica: entiende el significado. Inteligente: utiliza conocimiento estructurado."
    )
    
    # Campo de búsqueda y botón
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input("", placeholder="¿Qué deseas saber sobre cócteles? Ej: ¿Cómo preparar un Mojito?")
    
    with col2:
        search_button = st.button("Buscar")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Contenedor para mostrar resultados
with st.container():
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    # Mostrar resultados cuando se presiona el botón
    if search_button and query:
        # Guardar búsqueda en historial
        st.session_state.search_history.append(query)
        
        with st.spinner("Buscando información..."):
            # Preparar el comando según el tipo de búsqueda
            if search_type == "Simple":
                command = ["python", "-m", "agents.main", "search", query, "--no-llm"]
            elif search_type == "Semántica":
                command = ["python", "-m", "agents.main", "search", query]
            else:  # Ontología (Inteligente)
                # Usamos el comando adecuado para consultar la ontología
                command = ["python", "-m", "agents.main", "ontology", "query", query]
            
            # Ejecutar la búsqueda
            result = run_command(command)
            
            if result:
                st.markdown('<div class="answer-text">', unsafe_allow_html=True)
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No se encontraron resultados para tu consulta. Intenta reformularla o utiliza otro método de búsqueda.")
    
    # Mensaje cuando no hay consulta
    elif not st.session_state.search_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <img src="https://em-content.zobj.net/thumbs/240/apple/325/cocktail-glass_1f378.png" width="80">
            <h3>¡Bienvenido al Asistente de Cócteles!</h3>
            <p>Escribe tu pregunta arriba para comenzar a descubrir el mundo de los cócteles.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Historial de búsquedas recientes (colapsado por defecto)
if st.session_state.search_history:
    with st.expander("Búsquedas recientes"):
        for i, past_query in enumerate(reversed(st.session_state.search_history[-5:])):
            st.markdown(f"{i+1}. {past_query}")

# Pie de página
st.markdown("---")
st.caption("© 2025 Cocktail Assistant | Sistema de Información sobre Cócteles")
