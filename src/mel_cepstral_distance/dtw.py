import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def align_1d_sequences_using_dtw(sequence_1: np.ndarray, sequence_2: np.ndarray):
  _, path = fastdtw(sequence_1, sequence_2)
  path_np = np.array(path)
  stretched_seq_1 = sequence_1[path_np[:, 0]]
  stretched_seq_2 = sequence_2[path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2


def align_2d_sequences_using_dtw(seq_1: np.ndarray, seq_2: np.ndarray):
  if seq_1.shape[0] != seq_2.shape[0]:
    raise ValueError("Both sequences must have the same number of features (rows).")
  _, path = fastdtw(seq_1.T, seq_2.T, dist=euclidean)
  path_np = np.array(path)
  stretched_seq_1 = seq_1[:, path_np[:, 0]]
  stretched_seq_2 = seq_2[:, path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2
