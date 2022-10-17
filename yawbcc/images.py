import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def central_pad_and_crop(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Returns a centering padded and cropped version of the image.

    Args:
        image:
            The image to pad and crop as numpy array.
        size:
            The output size in pixels, given as a (width, height) tuple.

    Returns:
        ndarray: The reshaped image.
    """
    # vertical padding
    vpad = max(size[1] - img.shape[0], 0)
    top, bottom = vpad // 2, vpad // 2 + vpad % 2

    # horizontal padding
    hpad = max(size[0] - img.shape[1], 0)
    left, right = hpad // 2, hpad // 2 + hpad % 2

    # pad image
    img2 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # vertical cropping
    vcrop = img2.shape[0] - size[1]
    top, bottom = vcrop // 2, img2.shape[0] - (vcrop // 2 + vcrop % 2)

    # horizontal cropping
    hcrop = img2.shape[1] - size[0]
    left, right = hcrop // 2, img2.shape[1] - (hcrop // 2 + hcrop % 2)

    # crop image
    return img2[top:bottom, left:right]


def show_image_with_hist(img: np.ndarray, height: int=4) -> plt.Figure:
    """Returns a centering padded and cropped version of the image.

    Args:
        image:
            The image to display.
        height:
            The height of figure.

    Returns:
        A matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(height*2, height),
                                   constrained_layout=True, gridspec_kw={'wspace': 0.05})

    ax2.tick_params(axis='y', left=False, labelleft=False, right='on', labelright='on')
    ax2.set_box_aspect(img.shape[0] / img.shape[1])

    for channel, color in enumerate(['blue', 'green', 'red']):
        hist = np.bincount(img[..., channel].flatten(), minlength=256)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = range(256)
        ax2.plot(x, hist, color=color, lw=0.8)
        ax2.fill_between(x, hist, color=color, alpha=0.3)

    return fig

