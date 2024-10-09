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
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
