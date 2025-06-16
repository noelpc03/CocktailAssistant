import streamlit as st
import subprocess
import json
import os
import time
import base64

# Función para convertir imagen a base64
def get_base64_encoded_image(image_path):
    """
    Carga una imagen y la codifica en base64 para mostrarla en HTML
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None

# Función para redimensionar imagen
def resize_image(image_path, max_width=120):
    """
    Carga una imagen desde la ruta especificada y la redimensiona
    para asegurar que no sea demasiado grande.
    """
    try:
        from PIL import Image
        import io
        
        # Abrir y redimensionar la imagen manteniendo la proporción
        img = Image.open(image_path)
        
        # Calcular el ratio para mantener la proporción
        width_percent = (max_width / float(img.size[0]))
        height = int((float(img.size[1]) * float(width_percent)))
        
        # Redimensionar
        img = img.resize((max_width, height), Image.LANCZOS)
        
        # Guardar en un buffer de memoria
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        # Devolver el contenido del buffer codificado en base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error al redimensionar la imagen: {e}")
        # Si ocurre un error, intentar devolver la imagen original
        return get_base64_encoded_image(image_path)

# Configuración de la página
st.set_page_config(
    page_title="Cocktail Assistant",
    page_icon="🍹",
    layout="wide",
)

# Estilo personalizado
st.markdown("""
<style>
    /* Reglas globales para imágenes */
    img {
        max-width: 100% !important;
    }
    
    img.logo-image {
        width: 120px !important;
        max-width: 120px !important;
        height: auto !important;
    }
    
    /* Estilos generales */
    .main {
        padding: 2rem;
        max-width: 100%;
    }
    
    /* Contenedor principal centrado */
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin: 0 auto;
        max-width: 800px;
        padding: 1rem;
    }
    
    /* Logo */
    .logo-container {
        margin: 0 auto 0.5rem auto; /* Márgenes automáticos horizontales para centrar */
        text-align: center;
        width: 100%;
        display: block; /* Cambiado de flex a block para mejor compatibilidad con centrado */
    }
    
    .logo-image {
        width: 120px !important; /* Aumentado a 120px según preferencia del usuario */
        height: auto !important;
        margin: 0 auto 0.5rem auto; /* Centrado mejorado */
        display: block; /* Para asegurar el centrado */
    }
    
    /* Encabezado y descripción */
    .cocktail-header {
        color: #1E3A8A;
        font-family: 'Georgia', serif;
        font-weight: 600;
        font-size: 2.5rem;
        margin: 0.3rem auto 0.1rem auto; /* Auto márgenes horizontales para centrado */
        text-align: center;
        width: 100%;
        padding: 0;
        display: block;
    }
    
    .app-description {
        font-size: 1rem;
        color: #555;
        margin: 0 auto 1.2rem auto;
        max-width: 100%; /* Usar el 100% del ancho disponible para centrado perfecto */
        text-align: center;
        width: 100%;
        padding: 0;
        display: block;
        line-height: 1.5; /* Mejor espaciado vertical */
    }
    
    /* Campo de búsqueda */
    .search-input {
        width: 100%;
        margin: 0.5rem 0 1.5rem 0;
        max-width: 700px;
    }
    
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        width: 100%;
    }
    
    /* Botón de búsqueda */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
        margin-top: 0.25rem;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Contenedor y formato de respuestas */
    .result-area {
        margin-top: 1rem;
        width: 100%;
        background: transparent;
    }
    
    .response-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9f2ff 100%);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1E88E5;
        margin-top: 1rem;
        padding: 1.2rem;
        width: 100%;
    }
    
    .answer-text {
        font-size: 16px;
        line-height: 1.4;
        color: #333;
        text-align: left;
    }
    
    /* Eliminar fondos y bordes extras de Streamlit */
    .stAlert {
        border: none !important;
        background-color: transparent !important;
    }
    
    div.stMarkdown {
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Eliminar bordes y fondos adicionales */
    .element-container {
        margin: 0 !important;
    }
    
    /* Ajustar el espacio de los elementos Streamlit */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        margin: 0 !important;
    }
    
    /* Remover el espacio alrededor del markdown */
    .css-1544g2n.e1fqkh3o4 {
        padding: 0 !important;
    }
    
    /* Historial de búsquedas */
    .history-container {
        margin-top: 2rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
        width: 100%;
    }
    
    /* Pie de página */
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Ocultar elementos de Streamlit que no queremos mostrar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Función para ejecutar comandos del sistema
def run_command(command):
    try:
        # Aseguramos que el script tiene permisos de ejecución
        if command[0].endswith(".sh"):
            import os
            script_path = command[0]
            if os.path.exists(script_path) and not os.access(script_path, os.X_OK):
                subprocess.run(["chmod", "+x", script_path], check=True)
                print(f"Añadidos permisos de ejecución a {script_path}")
            elif not os.path.exists(script_path):
                print(f"ERROR: El script no existe en la ruta {script_path}")
                return f"Error: No se pudo encontrar el script del sistema en {script_path}. Verifica la instalación."
        
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
if 'previous_query' not in st.session_state:
    st.session_state.previous_query = ""

# Contenedor principal centrado
st.markdown('<div class="centered-content">', unsafe_allow_html=True)

# Logo, título y descripción centrados en un único contenedor
# Usando un único contenedor con CSS inline para controlar perfectamente el centrado
st.markdown('<div style="text-align: center; width: 100%; display: block;">', unsafe_allow_html=True)

# Logo
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    # Usamos la versión redimensionada de la imagen (tamaño de 120px de ancho)
    st.markdown(f'''
        <div class="logo-container">
            <img src="data:image/png;base64,{resize_image(logo_path, max_width=120)}" class="logo-image" alt="Cocktail Assistant Logo" style="width: 120px !important; height: auto !important;">
        </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown('<div class="logo-container"><div style="font-size: 3rem; text-align: center;">🍹</div></div>', unsafe_allow_html=True)

# Título y descripción en un mismo div para asegurar alineación exacta
st.markdown('''
<div style="text-align: center; width: 100%;">
    <h1 class="cocktail-header">Cocktail Assistant</h1>
    <p class="app-description">Tu asistente personal de cócteles</p>
</div>
''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Campo de búsqueda - agregamos espacio superior
st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="search-input">', unsafe_allow_html=True)
query = st.text_input("Consulta", 
                     placeholder="¿Qué deseas saber sobre cócteles? Ej: ¿Cómo preparar un Mojito?",
                     label_visibility="collapsed",
                     key="search_query")  # Agregamos una key para poder detectar cambios
search_button = st.button("Buscar")
st.markdown('</div>', unsafe_allow_html=True)

# Sección de resultados
st.markdown('<div class="result-area">', unsafe_allow_html=True)

# Variable para detectar si se presionó Enter en el campo de búsqueda
enter_pressed = query != st.session_state.get("previous_query", "")
if "previous_query" not in st.session_state:
    st.session_state.previous_query = query

# Mostrar resultados cuando se presiona el botón O se presiona Enter
if (search_button or enter_pressed) and query:
    # Guardar búsqueda en historial
    st.session_state.search_history.append(query)
    
    # Actualizar el estado para saber qué consulta se procesó
    st.session_state.previous_query = query
    
    with st.spinner("Buscando información..."):
        # Ajustar la ruta para encontrar el script desde la carpeta app
        # Usar ruta absoluta para evitar problemas con rutas relativas
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script_path = os.path.join(project_root, "agent_system.sh")
        command = [script_path, "search", query]
        
        # Ejecutar la búsqueda
        result = run_command(command)
        
        if result:
            # Procesar la salida para extraer solo la respuesta relevante
            clean_response = extract_relevant_response(result)
            
            # Mostrar la respuesta con un formato más compacto y limpio
            st.markdown(f'''
                <div class="response-container">
                    <div class="answer-text">
                        {clean_response}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            st.error(f"""No se pudieron obtener resultados. Posibles soluciones:
            
1. **Configuración del sistema**: Asegúrate de haber ejecutado estos comandos (desde la raíz del proyecto '{project_root}'):
   - `./agent_system.sh crawl` para recolectar información
   - `./agent_system.sh extract_ontology` para crear la ontología

2. **Problema de rutas**: La aplicación fue movida a la carpeta 'app'. Verifica que los scripts se ejecutan desde la ruta correcta.

3. **Reformula tu consulta**: Intenta con otra pregunta relacionada con cócteles.
            """)

st.markdown('</div>', unsafe_allow_html=True)

# Historial de búsquedas recientes (colapsado por defecto)
if st.session_state.search_history:
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    with st.expander("Búsquedas recientes"):
        for i, past_query in enumerate(reversed(st.session_state.search_history[-5:])):
            st.markdown(f"{i+1}. {past_query}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Cocktail Assistant | Sistema de Información sobre Cócteles", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
