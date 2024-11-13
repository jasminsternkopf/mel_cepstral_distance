from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_X_km,
                                               get_X_kn)
from mel_cepstral_distance.helper import fill_with_zeros_1d, fill_with_zeros_2d


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


def test_example_audio_sim():
  SIM_ORIG = Path("examples/similar_audios/original.wav")
  SIM_INF = Path("examples/similar_audios/inferred.wav")

  S1 = wavfile.read(SIM_ORIG)[1]
  S2 = wavfile.read(SIM_INF)[1]
  sample_rate = 22050

  # S1, S2 = fill_with_zeros_1d(S1, S2)

  n_fft = 512
  win_len = 512
  hop_length = win_len // 4
  window = "hanning"
  X_km_1 = get_X_km(S1, n_fft, win_len, hop_length, window)
  X_km_2 = get_X_km(S2, n_fft, win_len, hop_length, window)
  N = 40
  low_freq = 0
  high_freq = sample_rate / 2
  X_kn_1 = get_X_kn(X_km_1, sample_rate, N, n_fft, low_freq, high_freq)
  X_kn_2 = get_X_kn(X_km_2, sample_rate, N, n_fft, low_freq, high_freq)
  M = 12
  MC_X_ik = get_MC_X_ik(X_kn_1, M)
  MC_Y_ik = get_MC_X_ik(X_kn_2, M)
  # MC_X_ik, MC_Y_ik = align_mfccs_using_dtw(MC_X_ik, MC_Y_ik)
  s = 1
  D = 12
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  assert np.allclose(mean_mcd_over_all_k, 5.421872778503755)
