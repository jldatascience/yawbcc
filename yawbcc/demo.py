import tensorflow as tf
import numpy as np

def compute_grad_cam_heatmaps(images, conv_model, clf_model):
    with tf.device('/CPU:0'), tf.GradientTape() as tape:
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

