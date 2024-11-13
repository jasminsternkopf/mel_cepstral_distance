import numpy as np

from mel_cepstral_distance.computation import get_w_n_m, get_X_kn_fast


def test_mel_spectrogram_dimension_fix():
  X_km = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
  sample_rate = 8000
  N = 2
  n_fft = 8  # Anpassung von n_fft, um sicherzustellen, dass die Dimensionen übereinstimmen
  low_freq = 0
  high_freq = 4000
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  result = get_X_kn_fast(X_km, w_n_m)

  # Expected shape based on X_km shape and N
  expected_shape = (X_km.shape[0], N)
  assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}."
  assert np.allclose(result, np.array([
    [0.47712125, 1.2787536],
    [1.63346846, 1.99563519]
  ]))


def test_single_mel_band():
  X_km = np.array([[1, 2], [3, 4]])
  sample_rate = 16000
  N = 1
  n_fft = 2
  low_freq = 0
  high_freq = 8000
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  result = get_X_kn_fast(X_km, w_n_m)

  # Expected shape
  expected_shape = (X_km.shape[0], N)
  assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}."
  assert np.allclose(result, np.array([
    [9.64327467e-17],
    [0.95424251]
  ]))


def test_large_values_input():
  X_km = np.array([[1e10, 2e10, 3e10], [3e10, 4e10, 5e10]])
  sample_rate = 44100
  N = 3
  n_fft = 4
  low_freq = 0
  high_freq = 22050
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  result = get_X_kn_fast(X_km, w_n_m)

  # Ensure the shape is correct
  expected_shape = (X_km.shape[0], N)
  assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}."

  # Check for finite values
  assert np.all(np.isfinite(result)), "Expected all values to be finite."
  assert np.allclose(result, np.array([
    [-15.65355977, 20., 20.60205999],
    [-15.65355977, 20.95424251, 21.20411998]
  ]))


def test_zero_input():
  X_km = np.zeros((30, 3))
  sample_rate = 16000
  N = 2
  n_fft = 4
  low_freq = 0
  high_freq = 8000
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  result = get_X_kn_fast(X_km, w_n_m)

  # Check that the result is -inf due to log10 of zero input energy (with small eps added)
  assert np.all(result <= 0), "Expected all values to be <= 0 due to log10 of zero input energy."


def test_high_and_low_frequencies():
  X_km = np.random.rand(12, 5)
  sample_rate = 48000
  N = 4
  n_fft = 8
  low_freq = 100
  high_freq = 12000
  w_n_m = get_w_n_m(sample_rate, n_fft, N, low_freq, high_freq)
  result = get_X_kn_fast(X_km, w_n_m)

  # Expected shape
  expected_shape = (X_km.shape[0], N)
  assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}."
