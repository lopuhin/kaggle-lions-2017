from pathlib import Path

import cv2
import numpy as np


N_CLASSES = 5


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
