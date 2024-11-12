import numpy as np

from mel_cepstral_distance.computation import hz_to_mel


def test_hz_to_mel_zero_frequency():
  hz = 0
  expected = 0
  result = hz_to_mel(hz)
  assert result == expected, f"Expected {expected}, but got {result}."


def test_hz_to_mel_typical_value():
  hz = 1000
  expected = 2595 * np.log10(1 + 1000 / 700.0)  # Expected result using the formula
  result = hz_to_mel(hz)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_hz_to_mel_large_value():
  hz = 20000
  expected = 2595 * np.log10(1 + 20000 / 700.0)  # Expected result using the formula
  result = hz_to_mel(hz)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_hz_to_mel_small_value():
  hz = 10
  expected = 2595 * np.log10(1 + 10 / 700.0)  # Expected result using the formula
  result = hz_to_mel(hz)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_hz_to_mel_decimal_value():
  hz = 432.5
  expected = 2595 * np.log10(1 + 432.5 / 700.0)  # Expected result using the formula
  result = hz_to_mel(hz)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."
