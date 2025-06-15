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
        background-color: rgba(255, 255, 255, 0.7);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Nuevos estilos para mejorar la presentación de la respuesta */
    .response-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9f2ff 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #1E88E5;
    }
    .cocktail-image {
        text-align: center;
        margin-bottom: 1rem;
    }
    .cocktail-emoji {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Función para ejecutar comandos del sistema
def run_command(command):
    try:
        # Aseguramos que el script tiene permisos de ejecución
        if command[0].endswith(".sh"):
            import os
            script_path = command[0]
            if not os.access(script_path, os.X_OK):
                subprocess.run(["chmod", "+x", script_path], check=True)
                print(f"Añadidos permisos de ejecución a {script_path}")
        
        # Ejecutamos el comando pero sin verificar el código de retorno
        # para evitar errores por códigos de salida no cero
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            check=False  # Cambiado a False para no lanzar excepciones por códigos de salida
        )
        
        # Incluso si el comando falla, intentamos usar la salida que haya generado
        if result.stdout:
            print(f"Comando completado con código: {result.returncode}")
            return result.stdout
        else:
            print(f"El comando no produjo salida en stdout. Código de salida: {result.returncode}")
            if result.stderr:
                print(f"Error stderr: {result.stderr}")
                # Si hay información útil en stderr, la usamos
                return result.stderr
            return None
    except Exception as e:
        st.warning(f"Ocurrió un problema al procesar tu consulta, pero intentaremos continuar.")
        print(f"Excepción ejecutando el comando: {str(e)}")
        print(f"Comando intentado: {' '.join(command)}")
        return None

# Función para extraer solo la respuesta relevante de la salida del comando
def extract_relevant_response(output):
    """
    Extrae la respuesta generada del texto de salida completo,
    eliminando todos los mensajes de log y diagnóstico.
    """
    if not output:
        return "No se obtuvo respuesta del sistema. Verifica que el sistema esté configurado correctamente."

    # Imprimir la salida completa para diagnóstico
    print("\n===== SALIDA COMPLETA DEL COMANDO =====")
    print(output)
    print("===== FIN DE SALIDA DEL COMANDO =====\n")
    
    try:
        # Buscar la respuesta entre "Results for:" y "Command output saved"
        if "Results for:" in output:
            # Obtener todo lo que viene después de "Results for:"
            after_results = output.split("Results for:", 1)[1]
            
            # Saltar la línea que contiene la consulta
            if "\n" in after_results:
                response_with_extra = after_results.split("\n", 1)[1].strip()
                
                # Cortar en "Command output saved" si existe
                if "Command output saved" in response_with_extra:
                    clean_response = response_with_extra.split("Command output saved", 1)[0].strip()
                elif "QUERY RESULTS:" in response_with_extra:
                    clean_response = response_with_extra.split("QUERY RESULTS:", 1)[0].strip()
                else:
                    clean_response = response_with_extra
                
                # Eliminar líneas de guiones
                clean_response = '\n'.join([line for line in clean_response.split('\n') 
                                          if not line.strip().startswith('-') and 
                                             not line.strip() == ''])
                
                return clean_response.strip()
        
        # Si el método anterior falla, simplemente devolver la salida original
        # pero con un mensaje de identificación
        return output
        
    except Exception as e:
        print(f"Error al procesar la respuesta: {e}")
        # Como último recurso, devolver la salida original
        return output

# Inicializar estado de sesión
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Encabezado con logo y título
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🍹")
with col2:
    st.markdown('<h1 class="cocktail-header">Cocktail Assistant</h1>', unsafe_allow_html=True)

st.markdown('<p style="font-size: 18px;">Tu asistente personal de cocteles con inteligencia artificial. Pregunta cualquier cosa sobre recetas, ingredientes y más.</p>', unsafe_allow_html=True)
st.markdown("---")

# Contenedor para la interfaz de búsqueda
with st.container():
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Campo de búsqueda y botón
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input("Consulta", 
                             placeholder="¿Qué deseas saber sobre cócteles? Ej: ¿Cómo preparar un Mojito?",
                             label_visibility="collapsed")
    
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
        
        with st.spinner("Buscando información con IA..."):
            # Volvemos al comando básico que sí funcionaba
            command = ["./agent_system.sh", "search", query]
            
            # Ejecutar la búsqueda
            result = run_command(command)
            
            if result:
                # Procesar la salida para extraer solo la respuesta relevante
                clean_response = extract_relevant_response(result)
                
                # Mostrar la respuesta con un formato mejorado
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown('<div class="cocktail-image"><div class="cocktail-emoji">🍹</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="answer-text">', unsafe_allow_html=True)
                st.markdown(clean_response)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("""No se pudieron obtener resultados. Posibles soluciones:
                
1. **Configuración del sistema**: Asegúrate de haber ejecutado estos comandos antes:
   - `./agent_system.sh crawl` para recolectar información
   - `./agent_system.sh extract_ontology` para crear la ontología

2. **Reformula tu consulta**: Intenta con otra pregunta relacionada con cócteles.
                """)
    
    # Mensaje cuando no hay consulta
    elif not st.session_state.search_history:
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <img src="https://em-content.zobj.net/thumbs/240/apple/325/cocktail-glass_1f378.png" width="80">
            <h3>¡Bienvenido al Asistente de Cócteles!</h3>
            <p>Escribe tu pregunta arriba para comenzar a descubrir el mundo de los cócteles con inteligencia artificial.</p>
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
st.caption("© 2025 Cocktail Assistant | Sistema de Información sobre Cócteles con IA")
