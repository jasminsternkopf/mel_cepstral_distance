import unittest

import numpy as np

from mcd.mcd_computation import get_mcd_dtw_from_paths


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

    self.assertAlmostEqual(res_similar[0], 9.48651454)
    self.assertAlmostEqual(res_somewhat_similar[0], 10.37819387)
    self.assertAlmostEqual(res_unsimilar[0], 14.49857858)

    self.assertEqual(res_similar[1], 539)
    self.assertEqual(res_somewhat_similar[1], 953)
    self.assertEqual(res_unsimilar[1], 1027)


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
