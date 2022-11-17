"""
This script converts Munich dataset to one compatible to Barcelona dataset:
    - reorganize folders
    - rename files
    - convert to JPEG

Download original dataset from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080958
"""

import pandas as pd
import numpy as np
import pathlib
from PIL import Image
from tqdm import tqdm

BASE_DIR = pathlib.Path.home() / 'yawbcc_data'
SRC_DIR = BASE_DIR / 'PKG - AML-Cytomorphology'
DATA_DIR = BASE_DIR / 'munich'

# Mapping munich to barcelona
data = {'origin': ['BAS', 'EBO', 'EOS', 'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'PMO'],
        'label': ['BA', 'ERB', 'EO', 'LY', 'MMY', 'MO', 'MY', 'BNE', 'SNE', 'PMY'],
        'group': ['BASOPHIL', 'ERYTHROBLAST', 'EOSINOPHIL', 'LYMPHOCYTE',
                  'IG', 'MONOCYTE', 'IG', 'NEUTROPHIL', 'NEUTROPHIL', 'IG']}
df1 = pd.DataFrame(data)

# Read annotations from munich dataset
df = pd.read_csv(SRC_DIR / 'annotations.dat', sep=' ', header=None, names=['image', 'origin', 'label1', 'label2'])
df[['label1', 'label2']].isna().all(axis=1)  # remove unsure images

df2 = df[['image', 'origin']].merge(df1, on='origin')
df2['image'] = str(SRC_DIR) + '/AML-Cytomorphology/' + df2['image']
df2['path'] = str(DATA_DIR) + '/' + df2['group'].str.lower() + '/' + df2['label'] + '_' + df2['image'].str.extract(r'(\d{4})', expand=False) + '.jpg'

# Create target directories
for folder in df2['group'].unique():
    (DATA_DIR / folder.lower()).mkdir(exist_ok=True, parents=True)

for rec in tqdm(df2.itertuples(), total=len(df2)):
    Image.open(rec.image).convert('RGB').save(rec.path)
