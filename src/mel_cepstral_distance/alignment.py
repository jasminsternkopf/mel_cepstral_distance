from typing import Literal, Tuple

import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from mel_cepstral_distance.helper import fill_with_zeros_2d, get_penalty


def align_X_km(X_km_A: np.ndarray, X_km_B: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert aligning in ["dtw", "pad"]
  X_km_A, X_km_B, pen = align_frames_2d(X_km_A.T, X_km_B.T, aligning)
  X_km_A = X_km_A.T
  X_km_B = X_km_B.T
  return X_km_A, X_km_B, pen


def align_X_kn(X_kn_A: np.ndarray, X_kn_B: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert aligning in ["dtw", "pad"]
  X_kn_A, X_kn_B, pen = align_frames_2d(X_kn_A.T, X_kn_B.T, aligning)
  X_kn_A = X_kn_A.T
  X_kn_B = X_kn_B.T
  return X_kn_A, X_kn_B, pen


def align_MC(MC_X_ik: np.ndarray, MC_Y_ik: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert aligning in ["dtw", "pad"]
  return align_frames_2d(MC_X_ik, MC_Y_ik, aligning)


def align_frames_2d(seq1: np.ndarray, seq2: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert aligning in ["dtw", "pad"]
  assert seq1.shape[0] == seq2.shape[0]
  former_len_A = seq1.shape[1]
  former_len_B = seq2.shape[1]
  if aligning == "dtw":
    seq1, seq2 = align_2d_sequences_using_dtw(seq1, seq2)
  else:
    assert aligning == "pad"
    seq1, seq2 = fill_with_zeros_2d(seq1, seq2)

  assert seq1.shape[1] == seq2.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, seq1.shape[1])
  return seq1, seq2, penalty


def align_1d_sequences_using_dtw(sequence_1: np.ndarray, sequence_2: np.ndarray):
  _, path = fastdtw(sequence_1, sequence_2)
  path_np = np.array(path)
  stretched_seq_1 = sequence_1[path_np[:, 0]]
  stretched_seq_2 = sequence_2[path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2


def align_2d_sequences_using_dtw(seq_1: np.ndarray, seq_2: np.ndarray):
  assert seq_1.shape[0] == seq_2.shape[
    0], "both sequences must have the same number of features (rows)"
  _, path = fastdtw(seq_1.T, seq_2.T, dist=euclidean)
  path_np = np.array(path)
  stretched_seq_1 = seq_1[:, path_np[:, 0]]
  stretched_seq_2 = seq_2[:, path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2
