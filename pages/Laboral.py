# Adaptado de https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import random
import streamlit as st
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    #st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_llm():
    llm = Ollama(model="llama3.1", request_timeout=120.0)
    return llm

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Incrustar PDF en HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Mostrar el archivo
    st.markdown(pdf_display, unsafe_allow_html=True)

# Cargar CSS
local_css("./css/streamlit.css")

# Llamar directorios y extraer recursos
input_dir_path = '/teamspace/studios/this_studio/document_db/laboral'
main_body_logo = "./images/icon_logo.png"
sidebar_logo = "./images/full_logo.png"
st.logo(sidebar_logo, icon_image=main_body_logo)


# Se procesan documentos
try:    
    loader = SimpleDirectoryReader(
        input_dir = input_dir_path,
        required_exts=[".pdf"],
        recursive=True
    )
        
    docs = loader.load_data()

    # llamamos al llm y al modelo de embedding
    llm=load_llm()
    embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    
    # Indexamos los documentos cargados
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    # Creamos el motor de consultar
    Settings.llm = llm
    query_engine = index.as_query_engine(streaming=True)

    # ====== Plantilla para responder prompts ======
    qa_prompt_tmpl_str = (
    "Información de contexto se encuentra a continuación.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada la información de contexto anterior, por favor responde a la consulta de manera amable y conversacional. Asegúrate de entender que términos como 'Art', 'art' y 'artículo' se refieren al mismo concepto.\n"
    "Ejemplo de equivalencias:\n"
    "- 'Art' es igual a 'art'.\n"
    "- 'Art' también se refiere a 'artículo' o 'Artículo'.\n"
    "Si no sabes la respuesta, di '¡No lo sé!'.\n"
    "Consulta: {query_str}\n"
    "Respuesta: "
    )

    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    
    # Cargar el resto de opciones
    col1, col2 = st.columns([6, 1])
    with col1:
        st.header(f"Pregúntale a JuriChat!")
    with col2:
        st.button("Reiniciar ↺", on_click=reset_chat)

    # Informar que el chat está listo
    st.success("Chat Listo!")
    # display_pdf(file)
        
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()     

# Se comienza el historial de mensajes
if "messages" not in st.session_state:
    reset_chat()

# Si hay un historial, mostrar mensajes de este en reinicio de la app
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# Se acepta el input del usuario
if prompt := st.chat_input("¿Qué me puedes decir sobre...?"):
    # Se añade el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Mostrar consultas del usuario en el espacio correspondiente
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar y mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simular transmisión de respuestas y aplicar técnica de chunking en la respuesta
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # full_response = query_engine.query(prompt)

        message_placeholder.markdown(full_response)
        # st.session_state.context = ctx

    # Añadir la respuesta del asistente al historial
    st.session_state.messages.append({"role": "assistant", "content": full_response})