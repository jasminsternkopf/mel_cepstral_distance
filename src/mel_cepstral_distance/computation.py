from typing import Literal

import numpy as np

from mel_cepstral_distance.helper import get_hz_points


def get_average_MCD(MCD_k: np.ndarray) -> float:
  """" Calculates the average Mel Cepstral Distance (MCD) over all frames """
  assert len(MCD_k.shape) == 1, f"Expected 1D array, but got {MCD_k.shape}"
  assert np.all(MCD_k >= 0), f"Negative values in MCD_k: {MCD_k}"
  mean_mcd_over_all_k = np.mean(MCD_k)
  return mean_mcd_over_all_k


def get_MCD_k(MC_X_ik: np.ndarray, MC_Y_ik: np.ndarray, s: int, D: int) -> np.ndarray:
  """ Calculates the Mel Cepstral Distance (MCD) for each frame """
  assert MC_X_ik.shape == MC_Y_ik.shape
  assert 0 <= s < D
  K = MC_X_ik.shape[1]

  MCD_k = np.zeros(K)
  for k in range(K):
    diff_square_sum = 0
    for i in range(s, D):
      diff_square_sum += (MC_X_ik[i, k] - MC_Y_ik[i, k]) ** 2
    MCD_k[k] = np.sqrt(diff_square_sum)

  return MCD_k


def get_MCD_k_fast(MC_X_ik: np.ndarray, MC_Y_ik: np.ndarray, s: int, D: int) -> np.ndarray:
  """ Calculates the Mel Cepstral Distance (MCD) for each frame """
  assert MC_X_ik.shape == MC_Y_ik.shape
  assert 0 <= s < D
  MCD_k = np.linalg.norm(MC_X_ik[s:D, :] - MC_Y_ik[s:D, :], axis=0)
  return MCD_k


def get_MC_X_ik(X_kn: np.ndarray, M: int) -> np.ndarray:
  """" Calculates the mel cepstrum of the mel spectrogram """
  # K: total frame count
  # M: number of cepstral coefficients
  assert X_kn.ndim == 2, f"Expected a 2D array, but got {X_kn.ndim} dimensions"
  assert isinstance(M, int) and M > 0, "M must be a positive integer"
  assert M <= X_kn.shape[1], "M must be less than or equal to the number of mel bands (columns) in X_kn"
  K: int = X_kn.shape[0]
  MC_X_ik: np.ndarray = np.zeros((M, K))
  for i in range(1, M + 1):
    for k in range(K):
      tmp = [
        X_kn[k, n - 1] * np.cos(i * (n - 0.5) * np.pi / M)
        for n in range(1, M + 1)
      ]
      MC_X_ik[i - 1, k] = np.sum(tmp)
  return MC_X_ik


def get_MC_X_ik_fast(X_kn: np.ndarray, M: int) -> np.ndarray:
  """
  Calculates the mel cepstrum of the mel spectrogram
  returns mel cepstrum with shape (M, #frames)
  """
  # K: total frame count
  # M: number of cepstral coefficients
  assert X_kn.ndim == 2, f"Expected a 2D array, but got {X_kn.ndim} dimensions"
  assert isinstance(M, int) and M > 0, "M must be a positive integer"
  assert M <= X_kn.shape[1], "M must be less than or equal to the number of mel bands (columns) in X_kn"

  n = np.arange(1, M + 1)
  i = np.arange(1, M + 1).reshape(-1, 1)  # Reshape for broadcasting
  cos_terms = np.cos(i * (n - 0.5) * np.pi / M)  # Precompute cosine terms

  MC_X_ik = cos_terms @ X_kn[:, :M].T

  return MC_X_ik


def get_X_kn(X_km: np.ndarray, w_n_m: np.ndarray) -> np.ndarray:
  """Calculates the mel spectrogram of the spectrogram"""
  # N = n mels
  assert X_km.shape[1] == w_n_m.shape[1], f"Expected {w_n_m.shape[1]} columns, but got {X_km.shape[1]}"

  K = X_km.shape[0]
  N = w_n_m.shape[0]

  # same as np.dot(energy_spec, w_n_m.T)
  log_inner = np.zeros((K, N))
  for k in range(K):
    for n in range(w_n_m.shape[0]):
      log_inner[k, n] = np.sum(abs(X_km[k, :]) ** 2 * w_n_m[n, :])

  X_kn = np.log10(log_inner + np.finfo(float).eps)
  return X_kn


def get_X_kn_fast(X_km: np.ndarray, w_n_m: np.ndarray) -> np.ndarray:
  """
  Calculates the mel spectrogram of the spectrogram
  returns mel spectrogram with shape (#frames, N)
  """
  assert X_km.shape[1] == w_n_m.shape[1], f"Expected {w_n_m.shape[1]} columns, but got {X_km.shape[1]}"

  # Compute the mel spectrogram using matrix multiplication for efficiency
  energy_spec = abs(X_km) ** 2
  log_inner = energy_spec @ w_n_m.T  # Matrix multiplication

  X_kn = np.log10(log_inner + np.finfo(float).eps)
  return X_kn


def get_w_n_m(sample_rate: int, n_fft: int, N: int, low_freq: float, high_freq: float) -> np.ndarray:
  ''' calculates mel filterbank '''
  # N: number of mel bands
  assert sample_rate > 0
  assert N > 0
  assert n_fft > 0
  assert high_freq <= sample_rate / 2
  assert low_freq < high_freq
  assert low_freq >= 0

  hz_points = get_hz_points(low_freq, high_freq, N)

  # logger = getLogger(__name__)
  # logger.info(f"Frequency ranges of mel-bins ({low_freq}Hz - {high_freq}Hz):")
  # for i in range(N + 1):
  #   logger.info(f"Mel-band {i}: {hz_points[i]:.2f}Hz - {hz_points[i + 1]:.2f}Hz")

  bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

  w_n_m = np.zeros((N, int(n_fft / 2 + 1)))

  for n in range(1, N + 1):
    left = bins[n - 1]
    center = bins[n]
    right = bins[n + 1]

    w_n_m[n - 1, left:center] = (np.arange(left, center) - left) / (center - left)
    w_n_m[n - 1, center:right] = (right - np.arange(center, right)) / (right - center)

  return w_n_m


def get_X_km(S: np.ndarray, n_fft: int, win_len: int, hop_length: float, window: Literal["hamming", "hanning"]) -> np.ndarray:
  """ 
  Short-Time Fourier Transform (STFT) 
  returns magnitude spectrogram with shape (#frames, n_fft // 2 + 1)
  """
  assert window in ["hamming", "hanning"]
  K = len(S)
  windowed_frames = np.array([
    S[k:k + win_len]
    for k in range(0, K - win_len, hop_length)
  ])

  # padding or truncating the frames to n_fft
  if win_len < n_fft:
    windowed_frames = np.pad(windowed_frames, ((0, 0), (0, n_fft - win_len)), mode='constant')
    win_len = n_fft
  elif win_len > n_fft:
    windowed_frames = windowed_frames[:, :n_fft]
    win_len = n_fft

  if window == "hamming":
    win = np.hamming(win_len)
  elif window == "hanning":
    win = np.hanning(win_len)
  else:
    assert False, f"Unknown window function '{window}'"
  stft = np.fft.rfft(windowed_frames * win, n=n_fft)
  magnitude_spec = np.abs(stft)
  return magnitude_spec
