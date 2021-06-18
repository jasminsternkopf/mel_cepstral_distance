import librosa
import numpy as np
from mcd.mcd_computation import (get_mcd_between_mel_spectograms,
                                 get_mcd_between_wav_files)

# region use_dtw=True


def test_len_of_output():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

  assert len(res_similar) == 3


def test_compare_mcds_of_different_audio_pairs_with_each_other():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

  assert res_similar[0] < res_somewhat_similar[0]


def test_compare_mcds_of_different_audio_pairs_with_each_other2():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

  assert res_somewhat_similar[0] < res_unsimilar[0]

# region similar audios


def test_mcd_of_similar_audios():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

  assert round(abs(res_similar[0] - 8.613918022570173), 7) == 0


def test_penalty_of_similar_audios():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

  assert round(abs(res_similar[1] - 0.18923933209647492), 7) == 0


def test_frame_number_of_similar_audios():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")

  assert res_similar[2] == 539

# endregion

# region somewhat similar audios


def test_mcd_of_somewhat_similar_audios():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

  assert round(abs(res_somewhat_similar[0] - 9.621031769651019), 7) == 0


def test_penalty_of_somewhat_similar_audios():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

  assert round(abs(res_somewhat_similar[1] - 0.259453781512605), 7) == 0


def test_frame_number_of_somewhat_similar_audios():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav")

  assert res_somewhat_similar[2] == 952

# endregion

# region unsimilar_audios


def test_mcd_of_unsimilar_audios():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

  assert round(abs(res_unsimilar[0] - 13.983229819153072), 7) == 0


def test_penalty_of_unsimilar_audios():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

  assert round(abs(res_unsimilar[1] - 0.283743842364532), 7) == 0


def test_frame_number_of_unsimilar_audios():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav")

  assert round(abs(res_unsimilar[2] - 1015), 7) == 0

# endregion

# endregion

# region use_dtw=False


def test_len_of_output_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False
  )
  assert len(res_similar) == 3


def test_compare_mcds_of_different_audio_pairs_with_each_other_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

  assert res_similar[0] < res_somewhat_similar[0]


def test_mcd_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_similar[0] - 19.526543043605322), 7) == 0


def test_penalty_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_similar[1] - 0.11946050096339111), 7) == 0


def test_frame_number_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    "examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav", use_dtw=False)

  assert res_similar[2] == 519

# endregion

# region somewhat similar audios


def test_mcd_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_somewhat_similar[0] - 21.97334780846056), 7) == 0


def test_penalty_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_somewhat_similar[1] - 0.015568862275449069), 7) == 0


def test_frame_number_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    "examples/somewhat_similar_audios/original.wav", "examples/somewhat_similar_audios/inferred.wav", use_dtw=False)

  assert res_somewhat_similar[2] == 835

# endregion

# region unsimilar_audios


def test_mcd_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_unsimilar[0] - 19.473360173721225), 7) == 0


def test_penalty_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_unsimilar[1] - 0.10652173913043472), 7) == 0


def test_frame_number_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    "examples/unsimilar_audios/original.wav", "examples/unsimilar_audios/inferred.wav", use_dtw=False)

  assert round(abs(res_unsimilar[2] - 920), 7) == 0

# endregion

# endregion

# region get_mfccs_of_mel_spectogram


def test_mcd_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load("examples/similar_audios/original.wav", mono=True)
  audio_2, sr_2 = librosa.load("examples/similar_audios/inferred.wav", mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert round(abs(res[0] - 8.613918022570173), 7) == 0


def test_penalty_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load("examples/similar_audios/original.wav", mono=True)
  audio_2, sr_2 = librosa.load("examples/similar_audios/inferred.wav", mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert round(abs(res[1] - 0.18923933209647492), 7) == 0


def test_frame_number_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load("examples/similar_audios/original.wav", mono=True)
  audio_2, sr_2 = librosa.load("examples/similar_audios/inferred.wav", mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert res[2] == 539

# endregion
