import streamlit as st

# Configuration de la page
# Page setting
st.set_page_config(layout="wide")

# Importation de style.css
# style.css import


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

title_1, title_2, title_3 = st.columns([1, 3, 1])
with title_2:
    st.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>A propos</p>", unsafe_allow_html=True)

image1, image2, image3, image4 = st.columns([2, 3, 3, 2])
with image2:
    st.image("images/ds_logo.png", caption="DataScientest",
             use_column_width=True)
with image3:
    st.write(
        "Projet réalisé dans le cadre de la formation Data Scientist par [DataScientest](https://datascientest.com/)")
    st.markdown("""
    >+ Damien Corral | damien.corral@gmail.com
    >+ Anastasiya Trushko Perney | anastasia.trushko@gmail.com
    >+ Jérémy Lavergne | jeremy.lav2009@gmail.com
    >+ Jordan Porcu | jordan.porcu@gmail.com

    >+ Sous la supervision de Maxime - DataScientest

    >+ Git Hub : https://github.com/DataScientest-Studio/yawbcc
    >+ Git Hub #2 : https://github.com/corralien/yawbcc
    >+ Git Hub #3 : https://github.com/ixokarey/yawbcc_streamlit
    
    >+ Api : https://yawbcc.demain.xyz/api/v1/docs
    >+ Dash : https://yawbcc.demain.xyz/
    """)
