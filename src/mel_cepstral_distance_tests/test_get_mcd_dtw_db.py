from pathlib import Path

import numpy as np
from librosa import load

from mel_cepstral_distance.core import get_mcd_with_dtw
from mel_cepstral_distance.core_db import (get_mcd_dtw_new, get_mcd_new, get_mel_spectrogram,
                                           get_mfccs_of_mel_spectrogram, mcd_to_db)


def generate_noise(duration: float, sample_rate: int) -> np.ndarray:
  noise = np.random.normal(0, 0.1, int(sample_rate * duration))
  return noise


def modify_noise(original_noise: np.ndarray, factor: float) -> np.ndarray:
  modified_noise = original_noise + factor * np.random.normal(0, 0.1, original_noise.shape)
  return modified_noise


def test_main():
  sr = 22050
  np.random.seed(1)
  sig_a = generate_noise(1, sr)
  sig_b = modify_noise(sig_a, 0.001)
  sig_a_spec = get_mel_spectrogram(sig_a, sr)
  sig_b_spec = get_mel_spectrogram(sig_b, sr)
  sig_a_mfccs = get_mfccs_of_mel_spectrogram(sig_a_spec, 13, False)
  sig_b_mfccs = get_mfccs_of_mel_spectrogram(sig_b_spec, 13, False)
  mcd = get_mcd_new(sig_a_mfccs, sig_b_mfccs)


def test_wavs():
  path_a = Path("examples/similar_audios/inferred.wav")
  path_b = Path("examples/similar_audios/original.wav")

  path_a = Path("examples/groundtruth-vs-waveglow/gt-LJ002-0092.wav")
  path_b = Path("examples/groundtruth-vs-waveglow/wg-LJ002-0092.wav")

  sig_a, sr = load(path_a, sr=None, mono=True, res_type=None,
                   offset=0.0, duration=None, dtype=np.float32)

  sig_b, sr = load(path_b, sr=None, mono=True, res_type=None,
                   offset=0.0, duration=None, dtype=np.float32)

  sig_a_spec = get_mel_spectrogram(sig_a, sr)
  sig_b_spec = get_mel_spectrogram(sig_b, sr)
  sig_a_mfccs = get_mfccs_of_mel_spectrogram(sig_a_spec, 13, True, True)
  sig_b_mfccs = get_mfccs_of_mel_spectrogram(sig_b_spec, 13, True, True)
  mcd, _ = get_mcd_dtw_new(sig_a_mfccs, sig_b_mfccs)
  mcd_alt, _ = get_mcd_with_dtw(sig_a_mfccs, sig_b_mfccs)
  mcd_db = mcd_to_db(mcd)
  mcd_alt_db = mcd_to_db(mcd_alt)
  return mcd
