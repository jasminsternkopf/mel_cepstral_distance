import numpy as np

from mel_cepstral_distance.alignment import fill_with_zeros_2d


def test_fill_rest_with_zeros():
  mfccs_1 = np.array([[1, 2, 3], [4, 5, 6]])
  mfccs_2 = np.array([[7, 8], [9, 10]])

  padded_mfccs_1, padded_mfccs_2 = fill_with_zeros_2d(mfccs_1, mfccs_2)

  # Check if both arrays have the same shape
  assert padded_mfccs_1.shape == padded_mfccs_2.shape, "Arrays should have the same shape after padding"

  # Check if the original values are unchanged
  assert np.array_equal(padded_mfccs_1[:, :3],
                        mfccs_1), "Original values in mfccs_1 should remain unchanged"
  assert np.array_equal(padded_mfccs_2[:, :2],
                        mfccs_2), "Original values in mfccs_2 should remain unchanged"

  # Check if padding is filled with zeros
  assert np.all(padded_mfccs_1[:, 3:] == 0), "Padded values in mfccs_1 should be zeros"
  assert np.all(padded_mfccs_2[:, 2:] == 0), "Padded values in mfccs_2 should be zeros"


test_fill_rest_with_zeros()
