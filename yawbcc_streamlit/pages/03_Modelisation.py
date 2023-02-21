import streamlit as st

# Configuration de la page
# Page setting
st.set_page_config(layout="wide")

# Importation de style.css
# style.css import


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("css/styles.css")

model_list = ["VGG16", "ResNet50", "MobileNet", "Xception"]

# Titre
# Title
title_1, title_2, title_3 = st.columns([2, 3, 2])
with title_2:
    st.markdown("<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Modélisation</p>", unsafe_allow_html=True)
    st.write("Choisissez un modèle afin de consulter son architecture, son entraînement ou ses résultats.")
    selected_model = st.selectbox("Choisissez un modèle : ", model_list)
    st.markdown(
        f"<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>{selected_model}</p>", unsafe_allow_html=True)

select1, select2, select3, select4 = st.columns([4, 3, 3, 4])
selected_view = select2.radio("Séléctionnez l'étape du modèle à visionner", options=[
                              "Architecture", "Entraînement", "Résultats"])

# Sélection "Architecture"
# Selected "Architecture"
select_title1, select_title2, select_title3 = st.columns([2, 3, 2])
if selected_view == "Architecture":
    selected_view2 = select3.radio(
        "Sélectionnez l'élément à visionner", ["Résumé", "Structure"])

    if selected_view2 == "Résumé":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Résumé</p>", unsafe_allow_html=True)
        view1, view2, view3, view4 = st.columns([4, 3, 3, 4])
        view2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Sans fine-tuning</p>", unsafe_allow_html=True)
        view2.image(
            f"images/{selected_model}/summary_initial.png", use_column_width=True)

        view3.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Avec fine-tuning</p>", unsafe_allow_html=True)
        view3.image(
            f"images/{selected_model}/summary_ft.png", use_column_width=True)

    elif selected_view2 == "Structure":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Structure</p>", unsafe_allow_html=True)
        if selected_model == "VGG16":
            vgg1, vgg2, vgg3, vgg4, vgg5 = st.columns([2, 1, 1, 1, 2])
            vgg3.image(f"images/{selected_model}/arch.png",
                       use_column_width=True)
        else:
            select_title2.image(
                f"images/{selected_model}/arch.png", use_column_width=True)

# Sélection "Entraînement"
# Selected "Entraînement"
elif selected_view == "Entraînement":
    selected_view2 = select3.radio("Sélectionnez l'élément à visionner", [
                                   "Sans fine-tuning", "Avec fine-tuning"])
    if selected_view2 == "Sans fine-tuning":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Sans fine-tuning</p>", unsafe_allow_html=True)
        select_title2.image(
            f"images/{selected_model}/fit_1_10.png", use_column_width=True)
    elif selected_view2 == "Avec fine-tuning":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Avec fine-tuning</p>", unsafe_allow_html=True)
        select_title2.image(
            f"images/{selected_model}/fit_11_20.png", use_column_width=True)

# Sélection "Résultats"
# Selected "Résultats"
elif selected_view == "Résultats":
    selected_view2 = select3.radio("Sélectionnez l'élément à visionner", [
                                   "Courbe d'apprentissage", "Matrice de confusion", "Rapport de classification"])
    if selected_view2 == "Courbe d'apprentissage":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Courbe d'apprentissage</p>", unsafe_allow_html=True)
        select_title2.image(
            f"images/{selected_model}/acc_loss_plot.png", use_column_width=True)
    elif selected_view2 == "Matrice de confusion":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Matrice de confusion</p>", unsafe_allow_html=True)
        select_title2.image(
            f"images/{selected_model}/cm_heatmap.png", use_column_width=True)
    elif selected_view2 == "Rapport de classification":
        select_title2.markdown(
            "<p style='padding: 10px; border: 2px solid white;text-align: center;font-size: 20px;'>Rapport de classification</p>", unsafe_allow_html=True)
        select_title2.image(
            f"images/{selected_model}/cr.png", use_column_width=True)
