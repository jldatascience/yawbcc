"""
Create remapped barcelona dataset :
    - image size: 360x363
    - color: auto balanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from yawbcc.datasets import load_barcelona_wbc
from yawbcc.images import central_pad_and_crop

DATA_DIR = pathlib.Path.home() / 'yawbcc_data' / 'barcelona_remapped'

meta = load_barcelona_wbc()

meta[(meta['width'] != 360) & (meta['height'] != 363)]

for group in meta['group'].unique():
    (DATA_DIR / group.lower()).mkdir(parents=True, exist_ok=True)

for rec in tqdm(meta.itertuples(), total=len(meta)):
    img = cv2.imread(rec.path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = (img * gray.mean() / img.mean(axis=(0, 1))).astype(np.uint8)
    bgr = central_pad_and_crop(bgr, (360, 363))
    cv2.imwrite(str(DATA_DIR / rec.group.lower() / rec.image), bgr)
