import pickle
from pathlib import Path

import numpy as np
import pytest

from mel_cepstral_distance.api import get_amplitude_spectrogram, get_mel_spectrogram, get_mfccs
from mel_cepstral_distance.helper import samples_to_ms
from mel_cepstral_distance.silence import get_loudness_vals_X_kn

TEST_DIR = Path("src/mel_cepstral_distance_tests/api_tests")

AUDIO_A = TEST_DIR / "A.wav"

N_FFT = samples_to_ms(512, 22050)
SR = 22050


def get_X_kn():
  amp_spec = get_amplitude_spectrogram(AUDIO_A, sample_rate=SR, n_fft=N_FFT, window="hamming",
                                       hop_len=16, norm_audio=False, remove_silence=False, win_len=N_FFT)
  mel_spec = get_mel_spectrogram(
    amp_spec, SR, N_FFT, remove_silence=False, fmax=SR // 2, fmin=0, M=20)
  return mel_spec


def test_result_changes_after_silence_removal():
  res = get_mfccs(
    get_X_kn(),
    remove_silence=False,
  )

  assert res.shape == (20, 303)

  mean = get_loudness_vals_X_kn(get_X_kn()).mean()

  res = get_mfccs(
    get_X_kn(),
    remove_silence=True,
    silence_threshold=mean,
  )

  assert res.shape == (20, 173)


def test_no_silence_threshold_raises_error():
  with pytest.raises(ValueError):
    get_mfccs(get_X_kn(), remove_silence=True, silence_threshold=None)


def test_removing_silence_from_sig_too_hard_returns_empty():
  loudness_max = get_loudness_vals_X_kn(get_X_kn()).max()
  res = get_mfccs(
    get_X_kn(),
    remove_silence=True,
    silence_threshold=loudness_max + 1,
  )

  assert res.shape == (20, 0)


def test_one_dim_raises_error():
  with pytest.raises(ValueError):
    get_mfccs(
      np.zeros(20),
      remove_silence=False,
    )


def test_three_dim_raises_error():
  with pytest.raises(ValueError):
    get_mfccs(
      np.empty((303, 20, 10), dtype=np.float64),
      remove_silence=False,
    )


def test_zero_mel_bands_raises_error():
  with pytest.raises(ValueError):
    get_mfccs(
      np.empty((303, 0), dtype=np.float64),
      remove_silence=False,
    )


def test_empty_spec_returns_empty():
  empty_spec = np.empty((0, 20), dtype=np.float64)

  res = get_mfccs(
    empty_spec,
    remove_silence=False,
  )

  assert res.shape == (20, 0)


def create_outputs():
  # silence removal
  loudness_max = get_loudness_vals_X_kn(get_X_kn()).max()
  targets = [None, loudness_max / 2, loudness_max - 1, loudness_max + 1]

  outputs = []

  for sil_removal in targets:
    spec = get_mfccs(
      get_X_kn(),
      remove_silence=sil_removal is not None,
      silence_threshold=sil_removal,
    )
    outputs.append((sil_removal, spec))

  for vals in outputs:
    print("\t".join(str(i) if not isinstance(i, np.ndarray) else str(np.mean(i)) for i in vals))
  (TEST_DIR / "test_get_mfccs.pkl").write_bytes(pickle.dumps(outputs))


def test_outputs():
  outputs = pickle.loads((TEST_DIR / "test_get_mfccs.pkl").read_bytes())
  for sil_removal, expected_spec in outputs:
    spec = get_mfccs(
      get_X_kn(),
      remove_silence=sil_removal is not None,
      silence_threshold=sil_removal,
    )
    np.testing.assert_almost_equal(spec, expected_spec)


if __name__ == "__main__":
  create_outputs()
