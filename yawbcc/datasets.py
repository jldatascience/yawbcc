import pandas as pd
import requests
from PIL import Image
from zipfile import ZipFile
from pathlib import Path
from io import BytesIO
from glob import glob

DATA_ROOT_DIR = Path.home() / 'yawbcc_data'


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
    for filename in glob(str(DATA_DIR / '**/*.jpg')):
        file = Path(filename)
        img = Image.open(file)
        d = {'image': file.name,
             'group': file.parent.name.upper(),
             'label': file.stem.split('_')[0].upper(),
             'width': img.size[0],
             'height': img.size[1],
             'path': filename}
        data.append(d)

    # Create dataframe then post-process some columns
    df = pd.DataFrame(data)
    return df

