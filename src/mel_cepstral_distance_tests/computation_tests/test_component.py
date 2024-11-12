import numpy as np

from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_X_km,
                                               get_X_kn)


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
  N = 40
  low_freq = 0
  high_freq = sample_rate / 2
  X_kn_1 = get_X_kn(X_km_1, sample_rate, N, n_fft, low_freq, high_freq)
  X_kn_2 = get_X_kn(X_km_2, sample_rate, N, n_fft, low_freq, high_freq)
  M = 13
  MC_X_ik = get_MC_X_ik(X_kn_1, M)
  MC_Y_ik = get_MC_X_ik(X_kn_2, M)
  s = 1
  D = 12
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  assert np.allclose(mean_mcd_over_all_k, 4.259283236188423)
