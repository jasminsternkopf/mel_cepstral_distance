from typing import Any, Optional, Tuple

import numpy as np
import scipy.fftpack
from fastdtw.fastdtw import dtw
from librosa.feature import melspectrogram, mfcc
from scipy.fftpack import dct
from scipy.spatial.distance import euclidean


def get_mel_spectrogram(audio: np.ndarray, sr: int, *, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True, norm: Optional[Any] = None, dtype: np.dtype = np.float64):
  mel_spectrogram1 = melspectrogram(
    y=audio,
    sr=sr,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,  # 1 for energy, 2 for power
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )
  return mel_spectrogram1


def get_mfccs_of_mel_spectrogram(mel_spectrogram: np.ndarray, first_n_mfcc: int, take_log: bool, skip_zeroth: bool = True) -> np.ndarray:
  if take_log:
    mel_spectrogram = np.log10(mel_spectrogram + 1e-10)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
  #  Die Anzahl der MFCCs, die du erhalten kannst, entspricht der Anzahl der Mel-Bands, die in der Mel-Filterbank definiert sind.
  mfccs: np.ndarray = dct(mel_spectrogram, axis=-2, type=2, norm="ortho")
  mfccs = mfccs[:first_n_mfcc, :]

  # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth
  # coefficient is omitted
  if skip_zeroth:
    mfccs = mfccs[1:, :]

  # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's MFCC uses is 2 times
  # bigger than the one we want to use (which appears in Kubicheks paper)

  mfccs /= 2

  return mfccs


def get_mcd_dtw_new(mfccs_1: np.ndarray, mfccs_2: np.ndarray):
  _, path = dtw(mfccs_1.T, mfccs_2.T, dist=euclidean)

  mfccs_1_stretched = np.array([mfccs_1[:, i] for i, _ in path], mfccs_1.dtype).T
  mfccs_2_stretched = np.array([mfccs_2[:, j] for _, j in path], mfccs_2.dtype).T
  final_frame_number = len(path)
  return get_mcd_new(mfccs_1_stretched, mfccs_2_stretched), final_frame_number


def get_mcd_new(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> float:
  # mfccs shape: (n_mfccs, n_frames)
  # see: https://github.com/espnet/espnet/blob/7ba9035795df8de299fe3106889ae06fe285fa05/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_mcd.py#L169C9-L169C64
  assert mfccs_1.shape == mfccs_2.shape
  diff = np.subtract(mfccs_1, mfccs_2)
  diff_square = np.square(diff)
  diff_square_sum = np.sum(diff_square, axis=0)
  mcd = np.sqrt(2 * diff_square_sum)
  mcd_mean_all_frames = np.mean(mcd)

  return mcd_mean_all_frames


def mcd_to_db(mcd: float) -> float:
  mcd_db = (10 / np.log(10)) * mcd
  return mcd_db
