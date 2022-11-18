import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import pathlib

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam

from yawbcc.models import model_factory
from yawbcc.datasets import WBCDataSequence

# Segmented dataset
BASE_DIR = pathlib.Path.home() / 'yawbcc_data'
DATA_DIR = BASE_DIR / 'barcelona'

INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 32

TL_EPOCHS = 10
FT_EPOCHS = 10
NUM_TRAIN_LAYERS = 5

# Check GPU
print(f'Tensorflow: {tf.__version__}')
gpu_count = sum(1 for dev in tf.config.experimental.list_physical_devices()
                    if dev.device_type == 'GPU')
print(f'Number of GPUs: {gpu_count}')

# Find images of dataset
data = []
for file in DATA_DIR.glob('**/*.jpg'):
    d = {'image': file.name,
         'group': file.parent.name.upper(),
         'label': file.stem.split('_')[0].upper(),
         'path': file}
    data.append(d)

# For converting numeric class to label and vice versa
class_index = pd.read_csv('class_index.csv')
label_to_class = class_index.set_index('label').squeeze()
class_to_label = class_index.set_index('class').squeeze()

# Create dataframe and select columns
df = pd.DataFrame(data)
X = df['path']
y = df['group'].map(label_to_class)

# Split into 3 balanced datasets 
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=y, random_state=2022)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=2022)

# Remember each dataset
df.loc[X_train.index, 'dataset'] = 'train'
df.loc[X_test.index, 'dataset'] = 'test'
df.loc[X_valid.index, 'dataset'] = 'valid'

print(f'Train: {len(X_train)} records')
print(f'Valid: {len(X_valid)} records')
print(f'Test: {len(X_test)} records')

train_ds = WBCDataSequence(X_train, y_train, image_size=INPUT_SHAPE[:2], batch_size=BATCH_SIZE)
valid_ds = WBCDataSequence(X_valid, y_valid, image_size=INPUT_SHAPE[:2], batch_size=BATCH_SIZE)
test_ds = WBCDataSequence(X_test, y_test, image_size=INPUT_SHAPE[:2], batch_size=BATCH_SIZE)

# Modelisation
top_net = [
#    Flatten('channels_last', name='flatten'),
    Dense(256, activation='relu', name='fc1'),
    Dropout(0.2, seed=2022, name='dropout1'),
    Dense(256, activation='relu', name='fc2'),
    Dropout(0.2, seed=2022, name='dropout2'),
    Dense(len(np.unique(y)), activation='softmax', name='predictions')
]

transformers = [
#    RandomFlip(mode='horizontal', seed=2022, name='random_flip'),
#    RandomRotation(factor=(-0.2, 0.3), seed=2022, name='random_rotation')
]

# Step 1. Transfer Learning
app_name = 'VGG16'
model = model_factory(app_name, top_net, input_shape=INPUT_SHAPE, pooling='avg',
                      transformers=transformers, name=f'WBC-{app_name}')
model.summary()

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tl_history = model.fit(train_ds, epochs=TL_EPOCHS, validation_data=valid_ds)
model.save(DATA_DIR / f'{model.name.lower()}_tl.hdf5')

loss, accuracy = model.evaluate(test_ds)
print(f'Loss function: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Step 2. Fine tuning
idx = [idx for idx, layer in enumerate(model.layers) if layer.name.startswith(app_name.lower())][0]
app_model = model.layers[idx]
app_model.trainable = True
for layer in app_model.layers[:-NUM_TRAIN_LAYERS]:
    layer.trainable = False
model.summary()

adam = Adam(learning_rate=0.00001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ft_history = model.fit(train_ds, epochs=TL_EPOCHS+FT_EPOCHS, initial_epoch=TL_EPOCHS, validation_data=valid_ds)
model.save(DATA_DIR / f'{model.name.lower()}_ft.hdf5')

loss, accuracy = model.evaluate(test_ds)
print(f'Loss function: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
