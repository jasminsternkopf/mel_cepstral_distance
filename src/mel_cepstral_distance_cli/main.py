from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import numpy as np

from mel_cepstral_distance.mcd_computation import get_mcd_between_wav_files

WINDOWS = [
  "hamming",
  "hann",
  "boxcar",
  "triang",
  "blackman",
  "bartlett",
  "flattop",
  "parzen",
  "bohman",
  "blackmanharris",
  "nuttall",
  "barthann",
  "cosine",
  "exponential",
  "tukey",
  "taylor",
]


def init_mcd_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the MCD between two audio files."
  parser.add_argument("wav_1", type=Path, metavar="wav-1",
                      help="path to the first .wav file")
  parser.add_argument("wav_2", type=Path, metavar="wav-2",
                      help="path to the second .wav file")
  parser.add_argument("-f", "--n_fft", type=int, default=1024,
                      help="`n_fft/2+1` is the number of rows of the spectograms. `n_fft` should be a power of two to optimize the speed of the Fast Fourier Transformation")
  parser.add_argument("-l", "--hop_length", type=int, default=256)
  parser.add_argument("-w", "--window", type=str, choices=WINDOWS,
                      help="name of the window function; for details see: `scipy.signal.get_window`", default="hamming")
  parser.add_argument("-c", "--center", action="store_true")
  parser.add_argument("-m", "--n_mels", type=int, default=20)
  parser.add_argument("-t", "--htk", action="store_true")
  parser.add_argument("-o", "--norm", default=None)
  parser.add_argument("-y", "--dtype", type=np.dtype, default=np.float64)
  parser.add_argument("-n", "--n_mfcc", type=int, default=16)
  parser.add_argument("-d", "--use_dtw", action="store_true")
  return print_mcd_dtw_from_paths


def print_mcd_dtw_from_paths(wav_1: Path, wav_2: Path, n_fft: int, hop_length: int, window: str, center: bool, n_mels: int, htk: bool, norm, dtype: np.dtype, n_mfcc: int, use_dtw: bool):
  mcd, penalty, frames = get_mcd_between_wav_files(
    wav_file_1=wav_1,
    wav_file_2=wav_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    n_mfcc=n_mfcc,
    use_dtw=use_dtw
  )
  print(
    f"The mel-cepstral distance between the two WAV files is {mcd} and the penalty is {penalty}. This was computed using {frames} frames.")
