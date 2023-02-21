import streamlit as st

pages = st.source_util.get_pages('streamlit.py')
new_page_names = {
  'streamlit': 'Introduction',
  'DataViz' : 'Exploration & Visualisation des données',
  'Modelisation': 'Modelisation',
  'Preprocessing' : 'Preprocessing',
  'Application' : 'Application',
  'credits':'A propos'
}

for key, page in pages.items():
  if page['page_name'] in new_page_names:
    page['page_name'] = new_page_names[page['page_name']]
  
# Configuration de la page
# Page setting
st.set_page_config(page_title = "YAWBCC",
                  page_icon = "images/logo_test.png",
                  initial_sidebar_state="expanded",
                  layout="wide")

# Importation de style.css
# style.css import
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

# Titre 
# Title
title_1,title_2,title_3 = st.columns([2,3,2])
title_2.image("images/title_yawbcc.png",use_column_width=True)
# Sous titre 1 
# Subtitle #1
subtitle_1,subtitle_2,subtitle_3 = st.columns([2,3,2])
subtitle_2.markdown("<p style='padding: 20; border: 2px solid white;text-align: center;font-size: 50;'> Objectif </p>", unsafe_allow_html=True)

# Texte 1
# Text #1
subtitle_2.write("L’objectif de ce projet est de classifier les cellules sanguines en fonction de leurs caractéristiques morphologiques en utilisant des techniques d’apprentissage profond (ou Deep Learning). Pour y parvenir, nous utiliserons un des réseaux neuronaux de convolution (CNN) avec une méthode d’apprentissage par transfert (Transfer Learning) ainsi qu’une optimisation des poids des modèles grâce à la méthode du Fine-tuning.")
    
# Sous titre 2 
# Subtitle #2
subtitle_2.markdown("<p style='padding: 10; border: 2px solid white;text-align: center;font-size: 20;'>Contexte</p>", unsafe_allow_html=True)

# Image 1
# Image #1
imgcol1,imgcol2,imgcol3 = st.columns([2,1,2])
imgcol2.image("images/cell_types.png",caption = "Les différents types de cellules",use_column_width=True)


subtitle_1,subtitle_2,subtitle_3 = st.columns([2,3,2])
# Texte 2
# Text #2
subtitle_2.write("""
    Les diagnostics de la majorité des maladies hématologiques commencent par une analyse morphologique des cellules sanguines périphériques. Ces dernières circulent dans les vaisseaux sanguins et contiennent trois types de cellules principales suspendues dans du plasma :
             
    - l’érythrocyte (globule rouge)
    - le leucocyte (globule blanc)
    - le thrombocyte (plaquette) 

    Les leucocytes sont les acteurs majeurs dans la défense de l’organisme contre les infections. Ce sont des cellules nucléées qui sont elle-mêmes divisées en trois classes :
    
    - les granulocytes (subdivisés en neutrophiles segmentés et en bande, en éosinophiles et en basophiles)
    - les lymphocytes
    - les monocytes

    Lorsqu’un patient est en bonne santé, la proportion des différents types de globules blancs dans le plasma est d’environ 54-62% pour les granulocytes, 25-33% pour les lymphocytes et 3-10% pour les monocytes. Cependant,
    en cas de maladie, par exemple une infection ou une anémie régénérative, cette proportion est modifiée en même temps que le nombre total de globules blancs, et on peut trouver des granulocytes immatures (IG) (promyélocytes, myélocytes et métamyélocytes) ou des précurseurs érythroïdes, comme les érythroblastes. 
    
    """)