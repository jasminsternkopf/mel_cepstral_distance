import numpy as np

from mel_cepstral_distance.computation import get_average_MCD


def test_single_value():
  MCD_k = np.array([10])
  expected = 10
  result = get_average_MCD(MCD_k)
  assert result == expected, f"Expected {expected}, but got {result}."


def test_multiple_values():
  MCD_k = np.array([10, 20, 30])
  expected = 20
  result = get_average_MCD(MCD_k)
  assert result == expected, f"Expected {expected}, but got {result}."


def test_empty_array():
  MCD_k: np.ndarray = np.array([])
  result = get_average_MCD(MCD_k)
  assert np.isnan(result), f"Expected NaN for empty input, but got {result}."


def test_mixed_values():
  MCD_k = np.array([0, 20, 15])
  expected = 11.666666666666666
  result = get_average_MCD(MCD_k)
  assert result == expected, f"Expected {expected}, but got {result}."


def test_large_values():
  MCD_k = np.array([1e10, 2e10, 3e10])
  expected = 2e10
  result = get_average_MCD(MCD_k)
  assert result == expected, f"Expected {expected}, but got {result}."
