import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from zipfile import ZipFile
from pathlib import Path
from io import BytesIO
from .images import central_pad_and_crop


DATA_ROOT_DIR = Path.home() / 'yawbcc_data'


def load_wbc_dataset(dataset: str) -> pd.DataFrame:
    """Load WBC dataset.

    Returns:
        A dataframe with some metadata.

    .. _List of available datasets (or direct download):
       https://cloud.minesparis.psl.eu/index.php/s/hKwfHczrQYcLx0J
    """
    DATA_DIR = DATA_ROOT_DIR / dataset
    BASE_URL = 'https://cloud.minesparis.psl.eu/index.php/s/hKwfHczrQYcLx0J'

    if not DATA_ROOT_DIR.exists():
        DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Download dataset
    if not (DATA_DIR / 'dataset.csv').exists():
        with requests.get(f'{BASE_URL}/download?files={dataset}.zip', stream=True) as response:
            with ZipFile(BytesIO(response.content)) as archive:
                archive.extractall(DATA_ROOT_DIR)

    # Convert relative to absolute path according the system
    df = pd.read_csv(DATA_DIR / 'dataset.csv')
    df['path'] = [str(DATA_DIR / path) for path in df['path']]
    return df


def fetch_barcelona_wbc(force_download: bool=False) -> None:
    """Fetch WBC dataset from Barcelona.

    Args:
        force_download:
            Force to download again the dataset.
    """

    BARCELONA_DATASET = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/snkd93bnjr-1.zip'
    DATA_DIR = DATA_ROOT_DIR / 'barcelona'
    DATA_FILE = DATA_ROOT_DIR / 'PBC_dataset_normal_DIB.zip'

    # if archive file doesn't exist
    if not DATA_FILE.exists() or force_download:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Download dataset
        with requests.get(BARCELONA_DATASET, stream=True) as response:
            with ZipFile(BytesIO(response.content)) as archive:
                archive.extractall(DATA_ROOT_DIR)

        # Extract images (without leading directory)
        with ZipFile(DATA_FILE) as archive:
            for info in archive.infolist():
                parts = Path(info.filename).parts
                if parts[0].startswith(DATA_FILE.stem) and not parts[-1].startswith('.DS'):
                    if len(parts[1:]) > 1 and parts[1]:
                        info.filename = str(Path(*parts[1:]))
                        archive.extract(path=DATA_DIR, member=info)


def load_barcelona_wbc() -> pd.DataFrame:
    """Load WBC dataset from Barcelona.

    Returns:
        A dataframe with some metadata.
    """

    DATA_DIR = DATA_ROOT_DIR / 'barcelona'

    # Download dataset if needed
    fetch_barcelona_wbc()

    data = []
    # Extract simple informations from dataset
    for file in DATA_DIR.glob('**/*.jpg'):
        img = Image.open(file)
        d = {'image': file.name,
             'group': file.parent.name.upper(),
             'label': file.stem.split('_')[0].upper(),
             'width': img.size[0],
             'height': img.size[1],
             'path': str(file)}
        data.append(d)

    # Create dataframe then post-process some columns
    df = pd.DataFrame(data)
    return df


class WBCDataSequence(tf.keras.utils.Sequence):
    """
    Check documentation here: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, image_set, label_set, batch_size=32, image_size=(128, 128)):
        self.image_set = np.array(image_set)
        self.label_set = np.array(label_set)
        self.batch_size = batch_size
        self.image_size = image_size

    def __get_image(self, image):
        image = tf.keras.preprocessing.image.load_img(image, color_mode='rgb')
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = central_pad_and_crop(image_arr, self.image_size)
        return image_arr

    def __get_data(self, images, labels):
        image_batch = np.asarray([self.__get_image(img) for img in images])
        label_batch = np.asarray(labels)
        return image_batch, label_batch

    def __getitem__(self, index):
        images = self.image_set[index * self.batch_size:(index + 1) * self.batch_size]
        labels = self.label_set[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = self.__get_data(images, labels)
        return images, labels

    def __len__(self):
        return len(self.image_set) // self.batch_size + (len(self.image_set) % self.batch_size > 0)

