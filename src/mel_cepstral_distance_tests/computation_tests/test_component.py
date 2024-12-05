from logging import getLogger
from typing import Literal

import numpy as np
import numpy.typing as npt

from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_w_n_m,
                                               get_X_km, get_X_kn)
from mel_cepstral_distance.helper import get_hz_points


def norm_w_n_m(w_n_m: npt.NDArray, method: Literal["slaney", "sum"], hz_points: npt.NDArray) -> npt.NDArray:
  ''' normalizes the Mel filter bank '''
  assert method in ["slaney", "sum"]
  M, n_fft = w_n_m.shape
  if method == "slaney":
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    w_n_m *= enorm[:, np.newaxis]
  elif method == "sum":
    for n in range(M):
      sum_w = np.sum(w_n_m[n, :])
      if sum_w == 0:
        logger = getLogger(__name__)
        logger.warning(f"Mel band {n} has no energy")
      else:
        w_n_m[n, :] /= sum_w
  return w_n_m


def test_compontent():
  K = 10000
  np.random.seed(1)
  S1 = np.random.rand(K)
  S2 = np.random.rand(K)
  n_fft = 512
  win_len = 512
  hop_length = win_len // 2
  window = "hamming"
  X_km_1 = get_X_km(S1, n_fft, win_len, hop_length, window)
  X_km_2 = get_X_km(S2, n_fft, win_len, hop_length, window)
  sample_rate = 22050
  M = 40
  fmin = 0
  fmax = sample_rate / 2
  w_n_m = get_w_n_m(sample_rate, n_fft, M, fmin, fmax)
  assert w_n_m.shape[1] == X_km_1.shape[1]
  X_kn_1 = get_X_kn(X_km_1, w_n_m)
  X_kn_2 = get_X_kn(X_km_2, w_n_m)
  MC_X_ik = get_MC_X_ik(X_kn_1, M)
  MC_Y_ik = get_MC_X_ik(X_kn_2, M)
  s = 1
  D = 12
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  assert np.allclose(mean_mcd_over_all_k, 6.484722423858316)


def test_norm_filter_bank_does_not_change_result():
  K = 10000
  np.random.seed(1)
  S1 = np.random.rand(K)
  S2 = np.random.rand(K)
  n_fft = 512
  win_len = 512
  hop_length = win_len // 2
  window = "hamming"
  X_km_1 = get_X_km(S1, n_fft, win_len, hop_length, window)
  X_km_2 = get_X_km(S2, n_fft, win_len, hop_length, window)
  sample_rate = 22050
  M = 40
  fmin = 0
  fmax = sample_rate / 2
  w_n_m = get_w_n_m(sample_rate, n_fft, M, fmin, fmax)
  assert w_n_m.shape[1] == X_km_1.shape[1]
  for x in [None, "sum", "slaney"]:
    if x is None:
      w_n_m_normed = w_n_m
    else:
      w_n_m_normed = norm_w_n_m(w_n_m, x, get_hz_points(fmin, fmax, M))
    X_kn_1 = get_X_kn(X_km_1, w_n_m_normed)
    X_kn_2 = get_X_kn(X_km_2, w_n_m_normed)
    MC_X_ik = get_MC_X_ik(X_kn_1, M)
    MC_Y_ik = get_MC_X_ik(X_kn_2, M)
    s = 1
    D = 12
    MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
    mean_mcd_over_all_k = get_average_MCD(MCD_k)
    assert np.allclose(mean_mcd_over_all_k, 6.484722423858316)
