from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np
from fastdtw.fastdtw import fastdtw
from librosa.feature.spectral import mfcc
from scipy.spatial.distance import euclidean


@dataclass
class Params:
  n_fft: int = 1024
  hop_length: int = 256
  n_mels: int = 20
  no_of_coeffs_per_frame: int = 16
  use_dtw: bool = True
  window: str = "hamming"


@dataclass
class MCD_Result:
  mcd: float
  penalty: float
  final_frame_number: int
  added_frames: int


def get_mcd_dtw_from_paths(path_1: str, path_2: str, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming',
                           center: bool = False, n_mels: int = 20, htk: bool = True, norm=None, dtype=np.float64, no_of_coeffs_per_frame: int = 16) -> Tuple[float, int]:
  mel_spectogram_1, sr_1 = get_mel_spectogram_from_path(
    path_1, hop_length=hop_length, n_fft=n_fft, window=window, center=center, n_mels=n_mels, htk=htk, norm=norm, dtype=dtype)
  mel_spectogram_2, sr_2 = get_mel_spectogram_from_path(
    path_2, hop_length=hop_length, n_fft=n_fft, window=window, center=center, n_mels=n_mels, htk=htk, norm=norm, dtype=dtype)
  if sr_1 != sr_2:
    print("Warning: The sampling rates differ.")
  mfccs_1 = get_mfccs(mel_spectogram_1, no_of_coeffs_per_frame=no_of_coeffs_per_frame)
  mfccs_2 = get_mfccs(mel_spectogram_2, no_of_coeffs_per_frame=no_of_coeffs_per_frame)


def get_mel_spectogram_from_path(path: str, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming',
                                 center: bool = False, n_mels: int = 20, htk: bool = True, norm=None, dtype=np.float64) -> np.ndarray:
  audio, sr = librosa.load(path, mono=True)
  mel_spectogram = librosa.feature.melspectrogram(audio, sr, hop_length=hop_length, n_fft=n_fft, window=window,
                                                  center=center, n_mels=n_mels, htk=htk, norm=norm, dtype=dtype)
  return mel_spectogram, sr


def cos_func(i: int, n: int, n_mels: int) -> float:
  return np.cos((i + 1) * (n + 1 / 2) * np.pi / n_mels)


def get_mfccs(mel_spectogram: np.ndarray, no_of_coeffs_per_frame: int = 16) -> np.ndarray:
  log_mel_spectogram = np.log10(mel_spectogram)
  n_mels = mel_spectogram.shape[0]
  cos_matrix = np.fromfunction(cos_func, (no_of_coeffs_per_frame, n_mels),
                               dtype=np.float64, n_mels=n_mels)
  mfccs = cos_matrix @ log_mel_spectogram
  return mfccs


def mcd_with_dtw_and_penalty(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw=True) -> Tuple[float, float, int]:
  former_frame_number_1 = mfccs_1.shape[1]
  former_frame_number_2 = mfccs_2.shape[1]
  mcd, final_frame_number = mel_cepstral_dist_with_equaling_frame_number(
    mfccs_1, mfccs_2, use_dtw)
  penalty = dtw_penalty(former_frame_number_1,
                        former_frame_number_2, final_frame_number)
  return mcd, penalty, final_frame_number


def mel_cepstral_dist_with_equaling_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool) -> Tuple[float, int]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  equal_frame_number_mfcc_1, equal_frame_number_mfcc_2 = align_mfccs_with_dtw(mfccs_1.T, mfccs_2.T)
  return mel_cepstral_dist(equal_frame_number_mfcc_1, equal_frame_number_mfcc_2)


def equal_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool):
  if use_dtw:
    return align_mfccs_with_dtw(mfccs_1, mfccs_2)
  return fill_rest_with_zeros(mfccs_1, mfccs_2)


def align_mfccs_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  _, path_between_mfccs = fastdtw(mfccs_1, mfccs_2, dist=euclidean)
  path_for_input = list(map(lambda l: l[0], path_between_mfccs))
  path_for_output = list(map(lambda l: l[1], path_between_mfccs))
  mfccs_1 = mfccs_1[path_for_input]
  mfccs_2 = mfccs_2[path_for_output]
  return mfccs_1.T, mfccs_2.T


def fill_rest_with_zeros(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  frame_number_1 = mfccs_1.shape[1]
  frame_number_2 = mfccs_2.shape[1]
  diff = abs(frame_number_1 - frame_number_2)
  if diff > 0:
    adding_array = np.zeros(shape=(mfccs_1.shape[0], diff))
    if frame_number_1 < frame_number_2:
      mfccs_1 = np.concatenate((mfccs_1, adding_array), axis=1)
    else:
      mfccs_2 = np.concatenate((mfccs_2, adding_array), axis=1)
  assert mfccs_1.shape == mfccs_2.shape
  return mfccs_1, mfccs_2


def mel_cepstral_dist(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=0)
  mcd = np.mean(mfccs_diff_norms)
  frame_number = len(mfccs_diff_norms)
  return mcd, frame_number


def dtw_penalty(former_length_1: int, former_length_2: int, length_after_dtw: int) -> float:
  # lies between 0 and 1, the smaller the better
  return 2 - (former_length_1 + former_length_2) / length_after_dtw
