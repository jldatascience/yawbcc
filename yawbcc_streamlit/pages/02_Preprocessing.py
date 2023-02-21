import streamlit as st
from PIL import Image

# Configuration de la page
# Page setting
st.set_page_config(layout="wide")

# Importation de style.css
# style.css import


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("css/styles.css")

# Titre
# Title
title_1, title_2, title_3 = st.columns([4, 6, 4])
title_2.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Pré-processing</p>", unsafe_allow_html=True)

# Sous titre 1
# Subtitle #1


select_preprocessing = title_2.selectbox("Sélectionnez la méthode de ségmentation", [
                                         "Ségmentation par computer vision", "Ségmentation par deep learning"])

if select_preprocessing == "Ségmentation par computer vision":
    title_2.markdown(
        "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Méthode</p>", unsafe_allow_html=True)
    title_2.markdown("""
Nous utilisons la méthode suivante :
1. Charger une image du jeu de données
2. Convertir l’image en niveau de gris
3. Appliquer un filtre gaussien d’une taille de 5x5
4. Binariser l’image par seuillage (threshold)
5. Trouver les contours extérieurs (findContours)
6. Remplir les zones définies par les contours (drawContours)
7. Calculer la distance entre chaque pixel blanc et le pixel noir le plus proche (distanceTransform)
8. Trouver les coordonnées des maximums locaux à partir des distance précédentes (peak_local_max)
9. Segmenter les cellules présentes sur l’image selon l’algorithme
10. Conserver uniquement la cellule la plus au centre de l’image (regionprops)
Il ne reste plus qu’à récupérer le masque binaire pour isoler la cellule.
    """)
    with Image.open("images/segm_couleur.png") as img:
        title_2.image(img, use_column_width=True,
                      caption="Exemple des étapes de ségmentation")
    title_2.markdown(
        "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Résultat</p>", unsafe_allow_html=True)
    img1, img2, img3, img4 = st.columns([2, 1, 1, 2])
    with Image.open("images/MY_507649.jpg") as img:
        img2.image(img, use_column_width=True,
                   caption="Image originale (MY_507649.jpg)")
    with Image.open("images/comp.png") as img:
        img3.image(img, use_column_width=True,
                   caption="Image ségmentée par computer vision")

elif select_preprocessing == "Ségmentation par deep learning":
    title_2.markdown(
        "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Méthode</p>", unsafe_allow_html=True)
    title_2.markdown("""
    Nous essayons une méthode de segmentation basée sur le réseau spécialisé U-Net qui « se compose d’une partie contractante et une voie expansive, ce qui lui confère une architecture en forme
de «U». La partie contractante est un réseau de convolution typique qui consiste en une application
répétée de convolrutions, chacune suivie d’une unité linéaire rectifiée (ReLU) et d’une opération de
pooling maximum. Pendant la contraction, les informations spatiales sont réduites tandis que les
informations sur les caractéristiques sont augmentées. La voie expansive combine les informations
de caractéristiques géographiques et spatiales à travers une séquence de convolutions et concaténations ascendantes avec des fonctionnalités haute résolution issues de la voie contractante » (Wikipedia)
Nous utilisons U-Net dans sa version supervisée, c’est à dire que nous devons fournir au réseau une
image source et un masque binaire de la région d’intérêt (ROI) qui doit être extraite. Le réseau va
donc apprendre à repérer les ROI dans les images pour pouvoir les extraire plus tard par inférence.
En sortie, le modèle nous renvoie un masque que nous pourrons comparer au masque original.
Il ne reste plus qu’à récupérer le masque binaire pour isoler la cellule.""")
    with Image.open("images/unet_architecture.png") as img:
        title_2.image(img, use_column_width=True,
                      caption="Architecture du réseau U-Net")
    title_2.markdown(
        "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Résultat</p>", unsafe_allow_html=True)
    img1, img2, img3, img4 = st.columns([2, 1, 1, 2])
    with Image.open("images/MY_507649.jpg") as img:
        img2.image(img, use_column_width=True,
                   caption="Image originale (MY_507649.jpg)")
    with Image.open("images/deep.png") as img:
        img3.image(img, use_column_width=True,
                   caption="Image ségmentée par deep learning")
