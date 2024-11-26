import os
from pathlib import Path
from typing import Generator, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.signal import resample


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path
