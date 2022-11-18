from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
import inspect
import pathlib
import functools


def model_factory(app_name, top_net, input_shape=(128, 128, 3), pooling=None, transformers=None, name=None):
    # Try to locate a trained model based on its name
    model_function = getattr(tf.keras.applications, app_name)
    module_name = pathlib.Path(inspect.getfile(model_function)).stem
    model_module = getattr(tf.keras.applications, module_name)

    # Load base model and freeze layers
    app_model = model_function(weights='imagenet', input_shape=input_shape,
                               include_top=False, pooling=pooling)
    app_model.trainable = False

    # Input layer
    inputs = Input(shape=input_shape, name='input')

    # Image transformers
    transformers = [] if transformers is None else transformers
    x = functools.reduce(lambda x, f: f(x), transformers, inputs)

    # Base model preprocessing
    x = Lambda(model_module.preprocess_input, name='preprocess')(x)

    # Build model
    x = app_model(x, training=False)

    # Top net stack
    x = functools.reduce(lambda x, f: f(x), top_net, x)

    return Model(inputs, x, name=name)

