import numpy as np

from mel_cepstral_distance.computation import mel_to_hz


def test_mel_to_hz_zero():
  mel = 0
  expected = 0
  result = mel_to_hz(mel)
  assert result == expected, f"Expected {expected}, but got {result}."


def test_mel_to_hz_typical_value():
  mel = 1000
  expected = 700 * (10**(1000 / 2595) - 1)  # Expected result using the formula
  result = mel_to_hz(mel)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_mel_to_hz_large_value():
  mel = 4000
  expected = 700 * (10**(4000 / 2595) - 1)  # Expected result using the formula
  result = mel_to_hz(mel)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_mel_to_hz_small_value():
  mel = 10
  expected = 700 * (10**(10 / 2595) - 1)  # Expected result using the formula
  result = mel_to_hz(mel)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."


def test_mel_to_hz_decimal_value():
  mel = 432.5
  expected = 700 * (10**(432.5 / 2595) - 1)  # Expected result using the formula
  result = mel_to_hz(mel)
  assert np.isclose(result, expected), f"Expected {expected}, but got {result}."
