from pathlib import Path

from mel_cepstral_distance.api import compare_audio_files


def test_component():
  a = Path("examples/similar_audios/original.wav")
  b = Path("examples/similar_audios/inferred.wav")

  mcd, pen = compare_audio_files(
    a, b,
    align_target="spec",
    aligning="dtw",
  )
  assert mcd == 9.417554268936629
  assert pen == 0.2549019607843137
