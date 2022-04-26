from typing import Tuple

import numpy as np
from fastdtw.fastdtw import fastdtw
from librosa.feature import mfcc
from scipy.spatial.distance import euclidean

from mel_cepstral_distance.types import Frames, MelCepstralDistance, Penalty


def get_mfccs_of_mel_spectogram(mel_spectogram: np.ndarray, n_mfcc: int, take_log: bool) -> np.ndarray:
  mel_spectogram = np.log10(mel_spectogram) if take_log else mel_spectogram
  mfccs = mfcc(
    S=mel_spectogram,
    n_mfcc=n_mfcc + 1,
    norm=None,
    y=None,
    sr=None,
    dct_type=2,
    lifter=0,
  )

  # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth
  # coefficient is omitted
  # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's MFCC uses is 2 times
  # bigger than the one we want to use (which appears in Kubicheks paper)
  mfccs = mfccs[1:] / 2

  return mfccs


def get_mcd_and_penalty_and_final_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool
                                               ) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  former_frame_number_1 = mfccs_1.shape[1]
  former_frame_number_2 = mfccs_2.shape[1]
  mcd, final_frame_number = equal_frame_numbers_and_get_mcd(
    mfccs_1, mfccs_2, use_dtw)
  penalty = get_penalty(former_frame_number_1,
                        former_frame_number_2, final_frame_number)
  return mcd, penalty, final_frame_number


def equal_frame_numbers_and_get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray,
                                    use_dtw: bool) -> Tuple[MelCepstralDistance, Frames]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  if use_dtw:
    mcd, final_frame_number = get_mcd_with_dtw(mfccs_1, mfccs_2)
    return mcd, final_frame_number
  mfccs_1, mfccs_2 = fill_rest_with_zeros(mfccs_1, mfccs_2)
  mcd, final_frame_number = get_mcd(mfccs_1, mfccs_2)
  return mcd, final_frame_number


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


def get_mcd_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[MelCepstralDistance, Frames]:
  mfccs_1, mfccs_2 = mfccs_1.T, mfccs_2.T
  distance, path_between_mfccs = fastdtw(mfccs_1, mfccs_2, dist=euclidean)
  final_frame_number = len(path_between_mfccs)
  mcd = distance / final_frame_number
  return mcd, final_frame_number


def get_mcd(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[MelCepstralDistance, Frames]:
  assert mfccs_1.shape == mfccs_2.shape
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=0)
  mcd = np.mean(mfccs_diff_norms)
  frame_number = len(mfccs_diff_norms)
  return mcd, frame_number


def get_penalty(former_length_1: int, former_length_2: int, length_after_equaling: int) -> Penalty:
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_equaling
  return penalty
