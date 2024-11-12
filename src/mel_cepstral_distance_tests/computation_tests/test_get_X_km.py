import numpy as np

from mel_cepstral_distance.computation import get_X_km


def test_basic_stft_hamming():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 4
  win_len = 4
  hop_length = 2
  window = "hamming"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    np.abs(np.fft.rfft(np.hamming(win_len) * S[0:4], n=n_fft)),
    np.abs(np.fft.rfft(np.hamming(win_len) * S[2:6], n=n_fft))
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_basic_stft_hanning():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 4
  win_len = 4
  hop_length = 2
  window = "hanning"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    np.abs(np.fft.rfft(np.hanning(win_len) * S[0:4], n=n_fft)),
    np.abs(np.fft.rfft(np.hanning(win_len) * S[2:6], n=n_fft))
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_without_padding_hamming():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 4
  win_len = 4
  hop_length = 2
  window = "hamming"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [4.25, 2.54190873, 0.53],
    [7.65, 4.44883131, 0.53]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_with_padding_hamming():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 8
  win_len = 4
  hop_length = 2
  window = "hamming"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [6.33125099, 5.4737517, 3.79170493, 2.67243307, 2.31709321],
    [10.19125099, 8.55602393, 5.57246743, 3.8964064, 3.28765469]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_with_truncate_hamming():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 2
  win_len = 4
  hop_length = 2
  window = "hamming"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [0.24, 0.08],
    [0.56, 0.08]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_with_padding_hanning():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 8
  win_len = 4
  hop_length = 2
  window = "hanning"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [6.01222933, 5.36613998, 3.88539677, 2.66890073, 2.34466653],
    [9.51222933, 8.38661968, 5.81741845, 3.81142299, 3.39962467]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_with_truncation():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 4
  win_len = 4
  hop_length = 2
  window = "hanning"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [6.01222933, 5.36613998, 3.88539677, 2.66890073, 2.34466653],
    [9.51222933, 8.38661968, 5.81741845, 3.81142299, 3.39962467]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"
