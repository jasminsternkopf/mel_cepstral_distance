from pathlib import Path

import numpy as np
from scipy.io import wavfile

from mel_cepstral_distance.alignment import align_MC
from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_w_n_m,
                                               get_X_km, get_X_kn, norm_w_n_m)
from mel_cepstral_distance.helper import get_hz_points


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
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  assert w_n_m.shape[1] == X_km_1.shape[1]
  X_kn_1 = get_X_kn(X_km_1, w_n_m)
  X_kn_2 = get_X_kn(X_km_2, w_n_m)
  MC_X_ik = get_MC_X_ik(X_kn_1, N)
  MC_Y_ik = get_MC_X_ik(X_kn_2, N)
  s = 1
  D = 12
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  assert np.allclose(mean_mcd_over_all_k, 6.484722423858316)


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
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  # Normieren hat keinen Einfluss auf das Ergebnis
  w_n_m = norm_w_n_m(w_n_m, "sum", get_hz_points(low_freq, high_freq, N))
  X_kn_1 = get_X_kn(X_km_1, w_n_m)
  print(X_kn_1.mean())
  X_kn_2 = get_X_kn(X_km_2, w_n_m)
  MC_X_ik = get_MC_X_ik(X_kn_1, N)
  print(MC_X_ik.mean())
  MC_Y_ik = get_MC_X_ik(X_kn_2, N)
  MC_X_ik, MC_Y_ik, pen = align_MC(MC_X_ik, MC_Y_ik, aligning="dtw")
  s = 1
  D = 12
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  print(mean_mcd_over_all_k)
  assert np.allclose(mean_mcd_over_all_k, 16.564609750230623)


test_example_audio_sim()
