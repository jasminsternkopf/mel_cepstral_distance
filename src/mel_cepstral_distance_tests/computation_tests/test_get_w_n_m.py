import numpy as np

from mel_cepstral_distance.computation import get_w_n_m


def test_basic_mel_filterbank():
  sample_rate = 16000
  n_fft = 512
  N = 10
  fmin = 300
  fmax = 8000
  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)
  assert result.shape == (
    N, n_fft // 2 + 1), f"Expected shape {(N, n_fft // 2 + 1)}, but got {result.shape}."
  assert np.all(result >= 0), "Expected all filter values to be non-negative."


def test_single_mel_band():
  sample_rate = 16000
  n_fft = 256
  N = 1
  fmin = 0
  fmax = 4000
  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)
  assert result.shape == (
    N, n_fft // 2 + 1), f"Expected shape {(N, n_fft // 2 + 1)}, but got {result.shape}."
  assert np.sum(result) > 0, "Expected some non-zero values in the filter."


def test_fmaxuency_limit():
  sample_rate = 22050
  n_fft = 1024
  N = 5
  fmin = 100
  fmax = sample_rate / 2  # Maximum possible high frequency
  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)
  assert result.shape == (
    N, n_fft // 2 + 1), f"Expected shape {(N, n_fft // 2 + 1)}, but got {result.shape}."


def test_zero_fminuency():
  sample_rate = 44100
  n_fft = 1024
  N = 10
  fmin = 0
  fmax = 20000
  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)
  assert result.shape == (
    N, n_fft // 2 + 1), f"Expected shape {(N, n_fft // 2 + 1)}, but got {result.shape}."


def test_direct_array_comparison_nfft_sixteen():
  sample_rate = 8000
  n_fft = 16
  N = 2
  fmin = 0
  fmax = 4000
  expected = np.array([
    [0., 1., 0.5, 0., 0., 0., 0., 0., 0.],
    [0., 0., 0.5, 1., 0.8, 0.6, 0.4, 0.2, 0.]
  ])
  row_sums = expected.sum(axis=1, keepdims=True)
  expected = expected / row_sums

  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_direct_array_comparison_nfft_eight():
  sample_rate = 8000
  n_fft = 8
  N = 2
  fmin = 0
  fmax = 4000
  expected = np.array([
    [1., 0.5, 0., 0., 0.],
    [0., 0.5, 1., 0.5, 0.]
  ])
  row_sums = expected.sum(axis=1, keepdims=True)
  expected = expected / row_sums

  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"


def test_zero_sum_band():
  sample_rate = 16000
  N = 2
  n_fft = 4
  fmin = 0
  fmax = 8000
  result = get_w_n_m(sample_rate, n_fft, N, fmin, fmax)

  expected = np.array([
    [0., 0., 0.],
    [0.6666666, 0.3333333, 0.]
  ])

  assert result.shape == expected.shape, f"Expected shape {expected.shape}, but got {result.shape}."
  assert np.allclose(result, expected), f"Expected array:\n{expected}\nbut got:\n{result}"
