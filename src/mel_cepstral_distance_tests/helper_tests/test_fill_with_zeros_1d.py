import numpy as np

from mel_cepstral_distance.helper import fill_with_zeros_1d


def test_padding_with_different_lengths():
  array_1 = np.array([1, 2, 3])
  array_2 = np.array([4, 5])
  padded_array_1, padded_array_2 = fill_with_zeros_1d(array_1, array_2)

  assert len(padded_array_1) == len(
    padded_array_2), "Arrays should have the same length after padding"


def test_original_values_unchanged_after_padding():
  array_1 = np.array([1, 2, 3])
  array_2 = np.array([4, 5])
  padded_array_1, padded_array_2 = fill_with_zeros_1d(array_1, array_2)

  assert np.array_equal(
    padded_array_1[:3], array_1), "Original values in array_1 should remain unchanged"
  assert np.array_equal(
    padded_array_2[:2], array_2), "Original values in array_2 should remain unchanged"


def test_padding_filled_with_zeros():
  array_1 = np.array([1, 2, 3])
  array_2 = np.array([4, 5])
  padded_array_1, padded_array_2 = fill_with_zeros_1d(array_1, array_2)

  assert np.all(padded_array_1[3:] == 0), "Padded values in array_1 should be zeros"
  assert np.all(padded_array_2[2:] == 0), "Padded values in array_2 should be zeros"


def test_same_length_arrays_remain_unchanged():
  array_3 = np.array([7, 8, 9])
  array_4 = np.array([10, 11, 12])
  padded_array_3, padded_array_4 = fill_with_zeros_1d(array_3, array_4)

  assert np.array_equal(
    padded_array_3, array_3), "Arrays should remain unchanged if they have the same length"
  assert np.array_equal(
    padded_array_4, array_4), "Arrays should remain unchanged if they have the same length"


def test_empty_array_padding():
  array_5 = np.array([])
  array_6 = np.array([1, 2, 3])
  padded_array_5, padded_array_6 = fill_with_zeros_1d(array_5, array_6)

  assert len(padded_array_5) == len(
    padded_array_6), "Arrays should have the same length after padding"
  assert np.all(padded_array_5 == 0), "Empty array should be filled with zeros"
