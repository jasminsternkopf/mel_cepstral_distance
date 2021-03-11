import unittest

import numpy as np

from mcd.mcd_computation import get_mcd_and_penalty_and_frame_number_from_path


class UnitTests(unittest.TestCase):
  def __init__(self, methodName: str) -> None:
    super().__init__(methodName)

  # region use_dtw=True

  def test_len_of_output(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

    self.assertEqual(len(res_similar), 3)

  def test_compare_mcds_of_different_audio_pairs_with_each_other(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

    self.assertTrue(res_similar[0] < res_somewhat_similar[0])

  def test_compare_mcds_of_different_audio_pairs_with_each_other2(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertTrue(res_somewhat_similar[0] < res_unsimilar[0])

  # region similar audios

  def test_mcd_of_similar_audios(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

    self.assertAlmostEqual(res_similar[0], 8.613918022570173)

  def test_penalty_of_similar_audios(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

    self.assertAlmostEqual(res_similar[1], 0.18923933209647492)

  def test_frame_number_of_similar_audios(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

    self.assertEqual(res_similar[2], 539)

  # endregion

  # region somewhat similar audios

  def test_mcd_of_somewhat_similar_audios(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

    self.assertAlmostEqual(res_somewhat_similar[0], 9.621031769651019)

  def test_penalty_of_somewhat_similar_audios(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

    self.assertAlmostEqual(res_somewhat_similar[1], 0.259453781512605)

  def test_frame_number_of_somewhat_similar_audios(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

    self.assertEqual(res_somewhat_similar[2], 952)

  # endregion

  # region unsimilar_audios

  def test_mcd_of_unsimilar_audios(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertAlmostEqual(res_unsimilar[0], 13.983229819153072)

  def test_penalty_of_unsimilar_audios(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertAlmostEqual(res_unsimilar[1], 0.283743842364532)

  def test_frame_number_of_unsimilar_audios(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

    self.assertAlmostEqual(res_unsimilar[2], 1015)

  # endregion

  # endregion

  # region use_dtw=False

  def test_len_of_output_fill_rest_with_zeros(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False
    )
    self.assertEqual(len(res_similar), 3)

  def test_compare_mcds_of_different_audio_pairs_with_each_other_fill_rest_with_zeros(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

    self.assertTrue(res_similar[0] < res_somewhat_similar[0])

  # def test_compare_mcds_of_different_audio_pairs_with_each_other_fill_rest_with_zeros2(self):
  #   res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
  #     "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)
  #   res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
  #     "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

  #   self.assertTrue(res_somewhat_similar[0] < res_unsimilar[0])

  # region similar audios

  def test_mcd_of_similar_audios_fill_rest_with_zeros(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_similar[0], 19.526543043605322)

  def test_penalty_of_similar_audios_fill_rest_with_zeros(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_similar[1], 0.11946050096339111)

  def test_frame_number_of_similar_audios_fill_rest_with_zeros(self):
    res_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

    self.assertEqual(res_similar[2], 519)

  # endregion

  # region somewhat similar audios

  def test_mcd_of_somewhat_similar_audios_fill_rest_with_zeros(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_somewhat_similar[0], 21.97334780846056)

  def test_penalty_of_somewhat_similar_audios_fill_rest_with_zeros(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_somewhat_similar[1], 0.015568862275449069)

  def test_frame_number_of_somewhat_similar_audios_fill_rest_with_zeros(self):
    res_somewhat_similar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

    self.assertEqual(res_somewhat_similar[2], 835)

  # endregion

  # region unsimilar_audios

  def test_mcd_of_unsimilar_audios_fill_rest_with_zeros(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_unsimilar[0], 19.473360173721225)

  def test_penalty_of_unsimilar_audios_fill_rest_with_zeros(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_unsimilar[1], 0.10652173913043472)

  def test_frame_number_of_unsimilar_audios_fill_rest_with_zeros(self):
    res_unsimilar = get_mcd_and_penalty_and_frame_number_from_path(
      "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

    self.assertAlmostEqual(res_unsimilar[2], 920)

  # endregion

  # endregion


if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
