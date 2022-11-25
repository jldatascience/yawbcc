import tensorflow as tf
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops


def compute_grad_cam_heatmaps(images, conv_model, clf_model):
    with tf.device('/CPU:0'):
        with tf.GradientTape() as tape:
            # Get the output of convolution model (x)
            sources = conv_model(images)

            # Get the prediction from the output of convolution model (y)
            preds = clf_model(sources)
            indexes = tf.argmax(preds, axis=1)
            targets = tf.gather_nd(preds, tf.reshape(indexes, (-1, 1)), batch_dims=1)

        # Compute gradients
        grads = tape.gradient(targets, sources)

        # Compute the average of each filter (GlobalAveragePooling2D) (u, v, Z) -> Z
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # Create heatmaps
        heatmaps = tf.keras.backend.batch_dot(sources, pooled_grads)

        # Normalize heatmaps between 0 and 1
        original_shape = heatmaps.shape
        flatten_shape = heatmaps.shape[0], np.multiply(*heatmaps.shape[1:])
        m1 = tf.reshape(tf.maximum(heatmaps, 0), flatten_shape)
        m2 = tf.math.reduce_max(heatmaps, axis=(1, 2))
        heatmaps = tf.reshape(m1 / m2[..., None], original_shape)
        heatmaps = tf.floor(255 * heatmaps).numpy()

    return heatmaps


def color_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Trouve les contours extérieurs des cellules
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)# | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Crée un masque en remplissant les contours
    mask = cv2.drawContours(np.zeros(thresh.shape, np.uint8), contours, -1, 255, cv2.FILLED)

    # Distance entre chaque pixel blanc et le pixel noir le plus proche
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Trouve les maximas locaux (environ le nombre de cellules de l'image)
    coords = peak_local_max(dist, min_distance=10, labels=mask)

    # Crée les marqueurs à partir des coordonnées des maximas
    markers = np.zeros(dist.shape, dtype=np.uint8)
    markers[tuple(coords.T)] = 1
    markers = markers.cumsum().reshape(markers.shape) * markers

    # Algorithme watershed
    labels = watershed(-dist, markers, mask=mask)

    # Extrait le leucocyte par minimisation entre la distance du centre de la cellule et du centre de l'image
    cx, cy = dist.shape[1] // 2, dist.shape[0] // 2
    wbc = min(regionprops(labels), key=lambda x: np.sqrt((cx-x.centroid[1])**2 + (cy-x.centroid[0])**2))

    return np.uint8(labels==wbc.label)


def unet_segmentation(image, unet_cnn):
    with tf.device('/CPU:0'):
        mask = unet_cnn.predict(image[None]).squeeze()
    return np.uint8(mask >= 0.01)

