import streamlit as st

st.set_page_config(
    page_title="JuriChat",
)

st.write("# Bienvenidos/as a JuriChat! 👋")

# Llamar directorios y extraer recursos
input_dir_path = '/teamspace/studios/this_studio/document_db/familiar'
main_body_logo = "./images/icon_logo.png"
sidebar_logo = "./images/full_logo.png"
st.logo(sidebar_logo, icon_image=main_body_logo)


st.markdown(
    """
    Mucho gusto! Soy JuriChat, tu compañera en la búsqueda e investigación legislativa. 
    Mi objetivo es hacer que el acceso a leyes, artículos y documentos de El Salvador 
    sea más rápido y sencillo. Solo tienes que realizar una consulta, y yo te ayudaré 
    a encontrar la información precisa que necesitas o te proporcionaré una explicación 
    clara. Ya sea que estés investigando un caso, estudiando derecho o simplemente 
    aclarando dudas, estoy aquí para facilitar tu trabajo. ¡Hagamos que la búsqueda 
    legislativa sea más eficiente y accesible para todos! .
    
    ### ¿Qué módulos puedes encontrar conmigo ?
    - **Familia 👨‍👩‍👧‍👦**
    - **Laboral 💻​**
    - **Penal ⚖️**

    Proximamente tendremos muchos más módulos para ampliar nuestro conocimiento!

    ### ¿Cómo usar JuriChat?
    ¡Es muy facil, más de lo que crees!
    En la pagina web de Juri podrás encontrar tres módulos: Familia, Laboral y Penal, 
    si tienes dudas relacionadas con leyes y documentación legislativa de El Salvador
    enfocada en una área en especifico, selecciona el módulo con el área de tu interes, 
    y veras como un mini chat de texto con Juri y listo, solo chatea con Juri y haz tus 
    consultas y Juri te responderas a todas tus dudas, verás lo facil que es resolver
    tus preguntas con JuriChat 🤖​💙​.
    
"""
)

image_path = "./images/full_logo.png"
st.image(image_path, use_column_width=True)

