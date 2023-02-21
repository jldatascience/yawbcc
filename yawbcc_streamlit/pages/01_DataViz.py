import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

# Configuration de la page
# Page setting
st.set_page_config(
    page_title="Exploration & Visualisation des données", layout="wide")

# Importation de style.css
# style.css import


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("css/styles.css")

# Importation du dataframe
# Dataframe import
df = pd.read_csv("predictions.csv")

# Titre
# Title
title_1, title_2, title_3 = st.columns([2, 3, 2])
title_2.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Exploration et visualisation des données</p>", unsafe_allow_html=True)

# Sous titre 1
# Subtitle #1
title_2.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Origine des données</p>", unsafe_allow_html=True)
title_2.write("""  Nous avons mené notre projet en partant d’un jeu de données contenant un total de 17 092 images
    de cellules normales individuelles, qui ont été acquises à l’aide de l’analyseur CellaVision DM96 dans
    le laboratoire central de **la clinique hospitalière de Barcelone** (https://www.sciencedirect.com/science/article/pii/S2352340920303681).

    L’ensemble de données est organisé
    en huit groupes suivants : neutrophiles, éosinophiles, basophiles, lymphocytes, monocytes, granu-
    locytes immatures (promyélocytes, myélocytes et métamyélocytes), érythroblastes et plaquettes ou
    thrombocytes.""")

# Sélection "Extrait du dataset"
# Selection "Dataset sample"
expander_1 = title_2.expander("Extrait du dataset")
with expander_1:
    st.dataframe(df.iloc[:5, :5], use_container_width=True)

# Sélection "Distribution des classes"
# Selection "Classes distribution"
expander_2 = title_2.expander("Distribution des classes")
with expander_2:
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    (ax1, ax2) = gs.subplots(sharey='row')
    ax1.set_title("Avec 8 classes")
    sns.countplot(x=df.group.apply(lambda x: x[:2]), ax=ax1)
    ax2.set_title("Avec 13 classes")
    sns.countplot(x=df['label'].apply(lambda x: x[:3]),
                  hue=df['group'], dodge=False, ax=ax2)
    ax2.set_yticks([])
    plt.ylabel(None)
    plt.legend(loc='best', prop={'size': 12})
    st.pyplot(fig, clear_figure=True)

# Sélection "Image pour chacune des classes"
# Selection "Image for each class"
expander_3 = title_2.expander("Image pour chacune des classes")
with expander_3:
    st.image("images/images_group.png", use_column_width=True)

# Sélection "Dimension des images"
# Selection "Images dimension"
expander_4 = title_2.expander("Dimension des images")
with expander_4:
    df_dict = {"Dimension": ["359x360", "360x363", "360x360", "360x361",
                             "361x360", "362x360", "366x369"], "Nombre": [1, 16639, 198, 2, 1, 1, 250]}
    df_tailles = pd.DataFrame.from_dict(df_dict)
    df_tailles["Proportion"] = df_tailles["Nombre"] * \
        100/df_tailles["Nombre"].sum()
    st.dataframe(df_tailles.sort_values("Proportion", axis=0,
                 ascending=False), use_container_width=True)

# Sous titre 2
# Subtitle #2
title_2.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Conclusions</p>", unsafe_allow_html=True)
title_2.markdown("""
    - Dans le jeu de données de 17 092 images, il y a 8 classes: "basophile","neutrophile","ig","monocyte","éosinophile","érythroblaste","lymphocyte","plaquette".
    - La classe "neutrophile" comporte 3 types d'étiquettes : "BNE", "SNE" et "Neutrophile"
    - La classe "ig" comporte 4 types d'étiquettes : "MY", "PMY", "MMY", "IG".
    - Chaque autre classe n'a qu'un seul type d'étiquette.
    - Les images ont le même format.
    - Les images sont de tailles différentes :
         - 16 639 images d'une taille de 363x360 (hxw),
         - 250 images de 369x366,
         - 201 images de 360x360,
         - 2 images de 361x360, 
         - 1 image de 360x359, 
         - 1 image de 360x362, 
         - 1 image 360x361.""")
