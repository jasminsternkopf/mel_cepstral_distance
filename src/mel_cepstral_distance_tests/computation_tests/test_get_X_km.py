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
    np.fft.rfft(np.hamming(win_len) * S[0:4], n=n_fft),
    np.fft.rfft(np.hamming(win_len) * S[2:6], n=n_fft)
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
    np.fft.rfft(np.hanning(win_len) * S[0:4], n=n_fft),
    np.fft.rfft(np.hanning(win_len) * S[2:6], n=n_fft)
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
    [4.25 + 0.j, -2.23 - 1.22j, 0.53 + 0.j],
    [7.65 + 0.j, -3.61 - 2.6j, 0.53 + 0.j]
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
    [6.33125099 + 0.j, -2.26150868 - 4.9847303j,
        -1.84707889 + 3.31139333j, 2.42150868 - 1.13057253j,
        -2.31709321 + 0.j],
    [10.19125099 + 0.j, -3.09322734 - 7.97731095j,
        -2.97179815 + 4.71389531j, 3.57322734 - 1.55371466j,
        -3.28765469 + 0.j]
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
    [0.24 + 0.j, -0.08 + 0.j],
    [0.56 + 0.j, -0.08 + 0.j]
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
    [6.01222933 + 0.j, -2.42214304 - 4.78839027j,
        -1.8337814 + 3.42542754j, 2.42214304 - 1.12082747j,
        -2.34466653 + 0.j],
    [9.51222933 + 0.j, -3.5000981 - 7.6213321j,
        -3.05630233 + 4.94988621j, 3.5000981 - 1.50872743j,
        -3.39962467 + 0.j]
  ])

  assert result.shape == expected.shape == (
    2, 5), f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_stft_with_truncation():
  S = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  n_fft = 4
  win_len = 4
  hop_length = 2
  window = "hanning"
  result = get_X_km(S, n_fft, win_len, hop_length, window)

  expected = np.array([
    [3.75 + 0.j, -2.25 - 1.5j, 0.75 + 0.j],
    [6.75 + 0.j, -3.75 - 3.j, 0.75 + 0.j]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"
