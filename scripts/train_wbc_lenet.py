import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import pathlib
import cv2 as cv

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, GlobalAveragePooling2D 
from tensorflow.keras.optimizers import Adam

from yawbcc.models import model_factory
from yawbcc.datasets import WBCDataSequence

# Segmented dataset
BASE_DIR = pathlib.Path('/home/damien/yawbcc_data')
DATA_DIR = BASE_DIR / 'barcelona_remapped'
DATA_DIR = BASE_DIR / 'barcelona'

INPUT_SHAPE = (28, 28, 3)
BATCH_SIZE = 128

EPOCHS = 30

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

# Create dataframe and select columns
df = pd.DataFrame(data)
X = df['path']
y, cats = pd.factorize(df['group'])

# For converting numeric class to label
num_to_label = dict(enumerate(cats))

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
app_name = 'LeNet'

model = Sequential(name=f'WBC-{app_name}')
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='sigmoid', input_shape=INPUT_SHAPE))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid'))
model.add(MaxPool2D(strides=2))
#model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(84, activation='sigmoid'))
model.add(Dense(len(np.unique(y)), activation='softmax'))
model.summary()

adam = Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)
model.save(DATA_DIR / f'{model.name}.hdf5')

loss, accuracy = model.evaluate(test_ds)
print(f'Loss function: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
