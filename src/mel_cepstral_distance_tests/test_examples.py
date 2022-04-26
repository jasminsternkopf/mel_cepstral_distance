from pathlib import Path

from mel_cepstral_distance.mcd_computation import get_mcd_between_wav_files

SIM_ORIG = Path("examples/similar_audios/original.wav")
SIM_INF = Path("examples/similar_audios/inferred.wav")

SOSIM_ORIG = Path("examples/somewhat_similar_audios/original.wav")
SOSIM_INF = Path("examples/somewhat_similar_audios/inferred.wav")

DISSIM_ORIG = Path("examples/dissimilar_audios/original.wav")
DISSIM_INF = Path("examples/dissimilar_audios/inferred.wav")


def test_mcd_pen_frames_of_same_audio_with_dtw__returns_zero_zero_framecount():
  mcd, pen, frames = get_mcd_between_wav_files(SIM_ORIG, SIM_ORIG, use_dtw=True)

  assert mcd == 0.0
  assert pen == 0.0
  assert frames == 519


def test_mcd_pen_frames_of_similar_audios_with_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(SIM_ORIG, SIM_INF, use_dtw=True)

  assert mcd == 8.613918026817169
  assert pen == 0.18923933209647492
  assert frames == 539


def test_mcd_pen_frames_of_somewhat_similar_audios_with_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(SOSIM_ORIG, SOSIM_INF, use_dtw=True)

  assert mcd == 9.62103176737408
  assert pen == 0.259453781512605
  assert frames == 952


def test_mcd_pen_frames_of_dissimilar_audios_with_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(DISSIM_ORIG, DISSIM_INF, use_dtw=True)

  assert mcd == 13.983229820898327
  assert pen == 0.283743842364532
  assert frames == 1015


def test_mcd_pen_frames_of_same_audio_without_dtw__returns_zero_zero_framecount():
  mcd, pen, frames = get_mcd_between_wav_files(SIM_ORIG, SIM_ORIG, use_dtw=False)

  assert mcd == 0.0
  assert pen == 0.0
  assert frames == 519


def test_mcd_pen_frames_of_similar_audios_without_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(SIM_ORIG, SIM_INF, use_dtw=False)

  assert mcd == 19.526543043605322
  assert pen == 0.11946050096339111
  assert frames == 519


def test_mcd_pen_frames_of_somewhat_similar_audios_without_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(SOSIM_ORIG, SOSIM_INF, use_dtw=False)

  assert mcd == 21.97334780846056
  assert pen == 0.015568862275449069
  assert frames == 835


def test_mcd_pen_frames_of_dissimilar_audios_without_dtw():
  mcd, pen, frames = get_mcd_between_wav_files(DISSIM_ORIG, DISSIM_INF, use_dtw=False)

  assert mcd == 19.473360173721225
  assert pen == 0.10652173913043472
  assert frames == 920
