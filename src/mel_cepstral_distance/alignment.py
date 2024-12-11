from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from mel_cepstral_distance.helper import amp_to_mag


def align_X_km(X_km_A: npt.NDArray[np.complex128], X_km_B: npt.NDArray[np.complex128], aligning: Literal["dtw", "pad"], custom_radius: Optional[int]) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], float]:
  """
  Aligns two 2D sequences of complex numbers using either Dynamic Time Warping (DTW) or padding with zeros. For DTW, the sequences are temporarily converted to magnitude spectrograms.
  """
  assert aligning in ["dtw", "pad"]
  assert X_km_A.shape[1] == X_km_B.shape[1]
  former_len_A = X_km_A.shape[0]
  former_len_B = X_km_B.shape[0]
  paths: Optional[npt.NDArray] = None
  if aligning == "dtw":
    _, _, paths = align_2d_sequences_using_dtw(
      amp_to_mag(X_km_A).T,
      amp_to_mag(X_km_B).T,
      custom_radius,
    )
    X_km_A = X_km_A[paths[:, 0], :]
    X_km_B = X_km_B[paths[:, 1], :]
  else:
    assert aligning == "pad"
    X_km_A, X_km_B = fill_with_zeros_2d(X_km_A.T, X_km_B.T)
    X_km_A = X_km_A.T
    X_km_B = X_km_B.T

  assert X_km_A.shape[0] == X_km_B.shape[0]
  penalty = get_penalty(former_len_A, former_len_B, X_km_A.shape[0])
  return X_km_A, X_km_B, penalty


def align_X_kn(X_kn_A: npt.NDArray, X_kn_B: npt.NDArray, aligning: Literal["dtw", "pad"], custom_radius: Optional[int]) -> Tuple[npt.NDArray, npt.NDArray, float]:
  assert aligning in ["dtw", "pad"]
  X_kn_A, X_kn_B, pen = align_frames_2d(X_kn_A.T, X_kn_B.T, aligning, custom_radius)
  X_kn_A = X_kn_A.T
  X_kn_B = X_kn_B.T
  return X_kn_A, X_kn_B, pen


def align_MC(MC_X_ik: npt.NDArray, MC_Y_ik: npt.NDArray, aligning: Literal["dtw", "pad"], custom_radius: Optional[int]) -> Tuple[npt.NDArray, npt.NDArray, float]:
  assert aligning in ["dtw", "pad"]
  MC_X_ik, MC_Y_ik, pen = align_frames_2d(MC_X_ik, MC_Y_ik, aligning, custom_radius)
  return MC_X_ik, MC_Y_ik, pen


def align_MC_s_D(MC_X_ik: npt.NDArray, MC_Y_ik: npt.NDArray, s: int, D: int, aligning: Literal["dtw", "pad"], custom_radius: Optional[int]) -> Tuple[npt.NDArray, npt.NDArray, float]:
  assert MC_X_ik.shape[0] == MC_Y_ik.shape[0]
  M = MC_X_ik.shape[0]
  assert 0 <= s < D <= M
  assert aligning in ["dtw", "pad"]
  former_len_A = MC_X_ik.shape[1]
  former_len_B = MC_Y_ik.shape[1]
  if aligning == "dtw":
    _, _, paths = align_2d_sequences_using_dtw(MC_X_ik[s:D, :], MC_Y_ik[s:D, :], custom_radius)

    MC_X_ik = MC_X_ik[:, paths[:, 0]]
    MC_Y_ik = MC_Y_ik[:, paths[:, 1]]
  else:
    assert aligning == "pad"
    MC_X_ik, MC_Y_ik = fill_with_zeros_2d(MC_X_ik, MC_Y_ik)

  assert MC_X_ik.shape[1] == MC_Y_ik.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, MC_X_ik.shape[1])
  return MC_X_ik, MC_Y_ik, penalty


def align_frames_2d(seq1: npt.NDArray, seq2: npt.NDArray, aligning: Literal["dtw", "pad"], custom_radius: Optional[int]) -> Tuple[npt.NDArray, npt.NDArray, float]:
  assert aligning in ["dtw", "pad"]
  assert seq1.shape[0] == seq2.shape[0]
  former_len_A = seq1.shape[1]
  former_len_B = seq2.shape[1]
  if aligning == "dtw":
    seq1, seq2, _ = align_2d_sequences_using_dtw(seq1, seq2, custom_radius)
  else:
    assert aligning == "pad"
    seq1, seq2 = fill_with_zeros_2d(seq1, seq2)

  assert seq1.shape[1] == seq2.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, seq1.shape[1])
  return seq1, seq2, penalty


def align_2d_sequences_using_dtw(seq_1: npt.NDArray, seq_2: npt.NDArray, custom_radius: Optional[int] = None) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
  assert custom_radius is None or custom_radius >= 1
  assert seq_1.shape[0] == seq_2.shape[
    0], "both sequences must have the same number of features (rows)"
  assert seq_1.shape[1] > 0 and seq_2.shape[1] > 0, "both sequences must have at least one frame"
  if custom_radius is None:
    max_len = max(seq_1.shape[1], seq_2.shape[1])
  else:
    max_len = custom_radius
  _, path = fastdtw(seq_1.T, seq_2.T, dist=euclidean, radius=max_len)
  path_np = np.array(path)
  stretched_seq_1 = seq_1[:, path_np[:, 0]]
  stretched_seq_2 = seq_2[:, path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2, path_np


def get_penalty(former_length_1: int, former_length_2: int, length_after_equaling: int) -> float:
  assert former_length_1 > 0
  assert former_length_2 > 0
  assert length_after_equaling >= former_length_1
  assert length_after_equaling >= former_length_2
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_equaling
  return penalty


def fill_with_zeros_2d(array_1: npt.NDArray, array_2: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
  assert array_1.ndim == array_2.ndim == 2

  if array_1.shape[1] == array_2.shape[1]:
    return array_1, array_2

  max_frames = max(array_1.shape[1], array_2.shape[1])
  array_1 = np.pad(array_1, ((0, 0), (0, max_frames - array_1.shape[1])), mode='constant')
  array_2 = np.pad(array_2, ((0, 0), (0, max_frames - array_2.shape[1])), mode='constant')
  return array_1, array_2
