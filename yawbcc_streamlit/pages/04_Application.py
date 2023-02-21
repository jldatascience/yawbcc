import streamlit as st
import pathlib
from PIL import Image
import random
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from yawbcc.images import central_pad_and_crop
from yawbcc.demo import compute_grad_cam_heatmaps, color_segmentation, unet_segmentation
from yawbcc.datasets import load_wbc_dataset, WBCDataSequence
import tensorflow as tf
import cv2
import time

# Configuration de la page
# Page setting
st.set_page_config(layout="wide")

# Importation de style.css
# style.css import


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")

# Importation du dataframe et des images correspondantes
# Dataframe and corresponding image import
classes = ['BASOPHIL', 'EOSINOPHIL', 'ERYTHROBLAST', 'IG',
           'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'PLATELET']
idx_to_cls = dict(enumerate(classes))
cls_to_idx = {c: i for i, c in idx_to_cls.items()}
rng = np.random.default_rng(seed=2022)
# demodf = pd.read_csv("demo.csv") #local
# demodf["path"] = demodf["path"].apply(lambda x: f'classes/{x}')
df = load_wbc_dataset('barcelona')
demodf = df.groupby('group').sample(
    n=10, random_state=rng.bit_generator).sort_index()
demods = WBCDataSequence(demodf['path'], demodf['group'].map(
    cls_to_idx), image_size=(256, 256))
images = np.concatenate([batch[0] for batch in demods])
DATA_DIR = pathlib.Path.home() / 'yawbcc_data'
# Importation des modèles
# Models import
with tf.device('/CPU:0'):
    gc_conv = tf.keras.models.load_model(
        DATA_DIR / 'models' / 'gradcam_conv.hdf5')  # gradcam (conv)
    gc_clf = tf.keras.models.load_model(
        DATA_DIR / 'models' / 'gradcam_clf.hdf5')  # gradcam (clf)

with tf.device('/CPU:0'):
    unet_cnn = tf.keras.models.load_model(
        DATA_DIR / 'models' / 'unet_256.hdf5')  # unet

# Titre
# Title
title_1, title_2, title_3 = st.columns([2, 3, 2])
title_2.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Application</p>", unsafe_allow_html=True)
title_2.write(
    "Séléctionnez ou importez une image, et essayez les fonctions disponibles.")

# Sous titre
# Subtitle
text_1, text_2, text_3, text_4, text_5 = st.columns([2, 1, 1, 1, 2])

text_3.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Importation</p>", unsafe_allow_html=True)
text_2.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Aperçu de l'image</p>", unsafe_allow_html=True)
text_4.markdown(
    "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Sélection manuelle</p>", unsafe_allow_html=True)

# Titres des modes de sélection
# Mode select titles
# Bouton aléatoire
# Random button


def ma_fonction():
    st.session_state.search_class = random.choice(class_list)
    st.session_state.search_image = random.choice(np.array(image_list))

# Séléction manuelle + modification si utilisation du bouton aléatoire
# Manual selection + modifying if random button is used


with text_4:
    class_list = demodf["group"].unique()
    class_selected = st.selectbox('CLASSE', class_list, key='search_class')
    image_list = demodf[demodf["group"] == class_selected]["image"]
    image_selected = st.selectbox(
        'IMAGE', np.array(image_list), key='search_image')
    image_name = image_selected
    # alea_button = st.button("🎲",on_click=ma_fonction)
# Zone d'importation d'image
# Drag & drop
with text_3:
    dd_img = st.file_uploader(label="", label_visibility="hidden")

# Affichage de l'image en fonction de la sélection utilisée
# Image display according to selection mode
with text_2:
    if dd_img:
        image_selected = dd_img
        image_test = Image.open(dd_img).resize((256, 256))
        image_name = dd_img.name
        image_path = f'images_internet/{image_name}'
    else:
        image_path = demodf[demodf["image"] == image_selected]["path"].item()
        image_test = Image.open(image_path)
    st.image(image_test, caption=f"Image séléctionnée : {image_name}")

# Titre de boutons de fonctions
# Function buttons title
func_1, func_2, func_3 = st.columns([2, 3, 2])
func_2.markdown(
    "<p style='padding: 20px; border: 2px solid white;text-align: center;font-size: 20px;'>Fonctions</p>", unsafe_allow_html=True)

# Boutons de fonctions
# Function buttons
but_1, but_2, but_3, but_4, but_5 = st.columns([5, 1, 1, 1, 5])
with but_2:
    pred_button = st.button("Prediction")
with but_3:
    grad_button = st.button("Grad Cam")
with but_4:
    segm_button = st.button("Segmentation")

# Fonction pour la colormap du dataframe de prédiction
# Function used for the prediction dataframe colormap


def highlight_matrix(x):
    max_color = f"background-color: rgba({', '.join(str(int(c*255)) for c in cmap(x.max()))});"
    low_color = 'color: rgba(255, 255, 255, 0.2);'
    condlist = [x <= 0.1, x == x.max()]
    choicelist = [low_color, max_color]
    return np.select(condlist, choicelist, default=None)


# Prediction
if pred_button:
    pred_1, pred_2, pred_3 = st.columns([1, 2, 1])
    with pred_2:
        with st.spinner(""):
            images = [image_path]
            files = [('images', (image, open(image, 'rb')))
                     for image in images]
            df_proba = pd.DataFrame(
                columns=[x.upper() for x in demodf["group"].unique()])
            cmap = plt.cm.Greens  # Greens, Reds, Blues, jet, rainbow

            

            models_request = ''
            while models_request == '':
                try:
                    models_request = requests.get('https://yawbcc.demain.xyz/api/v1/models').json()
                    break
                except:
                    print("Connection refused by the server..")
                    print("Let me sleep for 5 seconds")
                    print("ZZzzzz...")
                    time.sleep(1)
                    print("Was a nice sleep, now let me continue...")
                    continue

        for model in models_request:
            files = [('images', (image, open(image, 'rb')))
                     for image in images]
            predict_proba_link = f"https://yawbcc.demain.xyz/api/v1/models/{model}/predict_proba"
            proba_request = requests.post(
                predict_proba_link, files=files).json()
            df_temp = pd.DataFrame.from_dict(
                proba_request).rename({0: model}, axis=0)
            df_proba = pd.concat([df_proba, df_temp])
        st.dataframe(df_proba.style.apply(
            highlight_matrix, axis=1), use_container_width=True)

# Gradcam
elif grad_button:
    col_grad1, col_grad2, col_grad3, col_grad4, col_grad5 = st.columns([
                                                                       3, 1, 1, 1, 3])
    if dd_img:
        col_grad3.write("Grad Cam non disponible.")
    else:
        with col_grad2:
            with st.spinner(""):
                idx = demodf[demodf["image"] == image_name].index.to_list()[0]
                image = np.uint8(images[demodf.index.get_loc(idx)])
                heatmap = compute_grad_cam_heatmaps(image[None], gc_conv, gc_clf)[
                    0].astype('uint8')
                cmap = plt.cm.get_cmap('jet')
                colors = np.uint8(255 * cmap(np.arange(256))[:, :3])
                colors[:30] = 0  # threshold low attention colors
                heatmap = cv2.resize(colors[heatmap], image.shape[:2])
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                gradcam = cv2.addWeighted(gray, 1, heatmap, 0.8, 0)
                st.image(image, caption="Image zoomée", use_column_width=True)
        with col_grad3:
            st.image(heatmap, caption="Heatmap générée", use_column_width=True)
        with col_grad4:
            st.image(gradcam, caption="Gradcam", use_column_width=True)

# Segmentation
elif segm_button:
    idx = demodf[demodf["image"] == image_name].index.to_list()[0]
    image = np.uint8(images[demodf.index.get_loc(idx)])
    cmask = color_segmentation(image)
    umask = unet_segmentation(image, unet_cnn)

    cimg = cv2.bitwise_and(image, image, mask=cmask)
    uimg = cv2.bitwise_and(image, image, mask=umask)

    col_seg1, col_seg2, col_seg3, col_seg4 = st.columns([3, 1, 1, 3])
    with col_seg2:
        st.image(cimg, caption="Segmentation par computer vision",
                 use_column_width=True)
    with col_seg3:
        st.image(uimg, caption="Segmentation par deep learning",
                 use_column_width=True)
