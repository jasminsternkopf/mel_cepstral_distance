import unittest

import numpy as np

from mcd.mcd_computation import (get_mcd_dtw_from_paths, mel_cepstral_dist_with_equaling_frame_number,
                                 mel_cepstral_dist_dtw)


class UnitTests(unittest.TestCase):
  def __init__(self, methodName: str) -> None:
    super().__init__(methodName)

  def test_compare_mcds_of_different_audio_pairs_with_each_other(self):
    res_similar = get_mcd_dtw_from_paths(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
    res_somewhat_similar = get_mcd_dtw_from_paths(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")
    res_unsimilar = get_mcd_dtw_from_paths(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertEqual(len(res_similar), 2)
    self.assertTrue(res_similar[0] < res_somewhat_similar[0])
    self.assertTrue(res_somewhat_similar[0] < res_unsimilar[0])

  def test_compare_mcds_of_different_audio_pairs_with_values(self):
    res_similar = get_mcd_dtw_from_paths(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
    res_somewhat_similar = get_mcd_dtw_from_paths(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")
    res_unsimilar = get_mcd_dtw_from_paths(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertAlmostEqual(res_similar[0], 8.613918022570173)
    self.assertAlmostEqual(res_somewhat_similar[0], 9.621031769651019)
    self.assertAlmostEqual(res_unsimilar[0], 13.983229819153072)

    self.assertEqual(res_similar[1], 539)
    self.assertEqual(res_somewhat_similar[1], 952)
    self.assertEqual(res_unsimilar[1], 1015)

  def test_shapes_do_not_fit__expect_exception(self):
    array_1 = np.array([[1, 2, 3], [4, 5, 6]])
    array_2 = np.array([[1, 2], [4, 5], [7, 8]])
    array_3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    with self.assertRaises(Exception):
      mel_cepstral_dist_dtw(array_1, array_2)

    with self.assertRaises(Exception):
      mel_cepstral_dist_with_equaling_frame_number(array_1, array_3)

    with self.assertRaises(Exception):
      mel_cepstral_dist_with_equaling_frame_number(array_2, array_3)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
