# Adaptado de https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os

import base64
import gc
import random
import tempfile
import time
import uuid

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

import streamlit as st

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    #st.session_state.file_cache = {}

session_id = st.session_state.id
client = None



@st.cache_resource
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


with st.sidebar:
    st.header(f"JurisChat")
    
    input_dir_path = '/teamspace/studios/this_studio/test-dir'

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
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, make sure to scan the context thoroughly, in case case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        
        # Informar que el chat está listo
        st.success("Chat Listo!")
        # display_pdf(file)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"JurisChat")

with col2:
    st.button("Reiniciar ↺", on_click=reset_chat)

# Se comienza el historial de mensajes
if "messages" not in st.session_state:
    reset_chat()

# Si hay un historial, mostrar mensajes de este en reinicio de la app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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