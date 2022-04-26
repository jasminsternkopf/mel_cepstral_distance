from pathlib import Path

import librosa
import numpy as np

from mel_cepstral_distance.mcd_computation import (get_mcd_between_mel_spectograms,
                                                   get_mcd_between_wav_files)

# region use_dtw=True

SIM_ORIG = Path("examples/similar_audios/original.wav")
SIM_INFE = Path("examples/similar_audios/inferred.wav")

SOSIM_ORIG = Path("examples/somewhat_similar_audios/original.wav")
SOSIM_INFE = Path("examples/somewhat_similar_audios/inferred.wav")

UNSIM_ORIG = Path("examples/unsimilar_audios/original.wav")
UNSIM_INFE = Path("examples/unsimilar_audios/inferred.wav")


def test_len_of_output():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE)

  assert len(res_similar) == 3


def test_mcd_pen_frames_of_similar_audios():
  mcd, pen, frames = get_mcd_between_wav_files(SIM_ORIG, SIM_INFE)

  assert mcd == 8.613918026817169
  assert pen == 0.18923933209647492
  assert frames == 539


def test_mcd_pen_frames_of_somewhat_similar_audios():
  mcd, pen, frames = get_mcd_between_wav_files(SOSIM_ORIG, SOSIM_INFE)

  assert mcd == 9.62103176737408
  assert pen == 0.259453781512605
  assert frames == 952


def test_mcd_pen_frames_of_unsimilar_audios():
  mcd, pen, frames = get_mcd_between_wav_files(UNSIM_ORIG, UNSIM_INFE)

  assert mcd == 13.983229820898327
  assert pen == 0.283743842364532
  assert frames == 1015


# region use_dtw=False


def test_len_of_output_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE, use_dtw=False
  )
  assert len(res_similar) == 3


def test_compare_mcds_of_different_audio_pairs_with_each_other_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE, use_dtw=False)
  res_somewhat_similar = get_mcd_between_wav_files(
    SOSIM_ORIG, SOSIM_INFE, use_dtw=False)

  assert res_similar[0] < res_somewhat_similar[0]


def test_mcd_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE, use_dtw=False)

  assert round(abs(res_similar[0] - 19.526543043605322), 7) == 0


def test_penalty_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE, use_dtw=False)

  assert round(abs(res_similar[1] - 0.11946050096339111), 7) == 0


def test_frame_number_of_similar_audios_fill_rest_with_zeros():
  res_similar = get_mcd_between_wav_files(
    SIM_ORIG, SIM_INFE, use_dtw=False)

  assert res_similar[2] == 519

# endregion

# region somewhat similar audios


def test_mcd_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    SOSIM_ORIG, SOSIM_INFE, use_dtw=False)

  assert round(abs(res_somewhat_similar[0] - 21.97334780846056), 7) == 0


def test_penalty_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    SOSIM_ORIG, SOSIM_INFE, use_dtw=False)

  assert round(abs(res_somewhat_similar[1] - 0.015568862275449069), 7) == 0


def test_frame_number_of_somewhat_similar_audios_fill_rest_with_zeros():
  res_somewhat_similar = get_mcd_between_wav_files(
    SOSIM_ORIG, SOSIM_INFE, use_dtw=False)

  assert res_somewhat_similar[2] == 835

# endregion

# region unsimilar_audios


def test_mcd_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    UNSIM_ORIG, UNSIM_INFE, use_dtw=False)

  assert round(abs(res_unsimilar[0] - 19.473360173721225), 7) == 0


def test_penalty_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    UNSIM_ORIG, UNSIM_INFE, use_dtw=False)

  assert round(abs(res_unsimilar[1] - 0.10652173913043472), 7) == 0


def test_frame_number_of_unsimilar_audios_fill_rest_with_zeros():
  res_unsimilar = get_mcd_between_wav_files(
    UNSIM_ORIG, UNSIM_INFE, use_dtw=False)

  assert round(abs(res_unsimilar[2] - 920), 7) == 0

# endregion

# endregion

# region get_mfccs_of_mel_spectogram


def test_mcd_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load(SIM_ORIG, mono=True)
  audio_2, sr_2 = librosa.load(SIM_INFE, mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert round(abs(res[0] - 8.613918022570173), 7) == 0


def test_penalty_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load(SIM_ORIG, mono=True)
  audio_2, sr_2 = librosa.load(SIM_INFE, mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert round(abs(res[1] - 0.18923933209647492), 7) == 0


def test_frame_number_of_mel_spectograms_of_similar_audios():
  audio_1, sr_1 = librosa.load(SIM_ORIG, mono=True)
  audio_2, sr_2 = librosa.load(SIM_INFE, mono=True)
  mel_1 = librosa.feature.melspectrogram(audio_1, sr=sr_1, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  mel_2 = librosa.feature.melspectrogram(audio_2, sr=sr_2, hop_length=256, n_fft=1024,
                                         window="hamming", center=False, n_mels=20, htk=True, norm=None, dtype=np.float64)
  res = get_mcd_between_mel_spectograms(mel_1, mel_2)

  assert res[2] == 539

# endregion
