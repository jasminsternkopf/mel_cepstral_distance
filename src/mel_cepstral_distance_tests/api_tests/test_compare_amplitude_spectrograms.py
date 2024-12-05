import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mel_cepstral_distance.api import compare_amplitude_spectrograms
from mel_cepstral_distance.computation import get_X_km
from mel_cepstral_distance.helper import get_n_fft_bins, norm_audio_signal, samples_to_ms

TEST_DIR = Path("src/mel_cepstral_distance_tests/api_tests")

AUDIO_A = TEST_DIR / "A.wav"
AUDIO_B = TEST_DIR / "B.wav"

SR = 22050
N_FFT = samples_to_ms(512, 22050)


def test_zero_dim_spec_raise_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), np.empty((305, 0)), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305, 0)), get_X_km_A(), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305, 0)), np.empty((305, 0)), SR, N_FFT)


def test_one_dim_spec_raise_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), np.empty((305)), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305)), get_X_km_A(), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305)), np.empty((305)), SR, N_FFT)


def test_three_dim_spec_raise_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), np.empty((305, 20, 10)), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305, 20, 10)), get_X_km_A(), SR, N_FFT)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(np.empty((305, 20, 10)), np.empty((305, 20, 10)), SR, N_FFT)


def get_X_km_A():
  sr1, signalA = wavfile.read(AUDIO_A)
  signalA = norm_audio_signal(signalA)
  return get_X_km(signalA, 512, 512, 128, "hamming")


def get_X_km_B():
  sr2, signalB = wavfile.read(AUDIO_B)
  signalB = norm_audio_signal(signalB)
  return get_X_km(signalB, 512, 512, 128, "hamming")


def test_aligning_with_pad_returns_same_for_spec_mel_mfcc():
  res = []
  for align_target in ["spec", "mel", "mfcc"]:
    mcd, pen = compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050), align_target=align_target, aligning="pad", dtw_radius=None,
    )
    res.append((mcd, pen))
  np.testing.assert_almost_equal(res[0], res[1])
  np.testing.assert_almost_equal(res[0], res[2])
  np.testing.assert_almost_equal(res[1], res[2])


def test_result_changes_after_silence_removal_before_padding_mel():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
    align_target="mel", aligning="pad", dtw_radius=None,
    remove_silence="spec", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
    align_target="mel", aligning="pad", dtw_radius=None,
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_result_changes_after_silence_removal_before_padding_mfcc():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
    align_target="mfcc", aligning="pad",
    remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2, dtw_radius=None,
  )
  mcd2, pen2 = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
    align_target="mfcc", aligning="pad",
    remove_silence="no",
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2, dtw_radius=None,
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_same_spec_returns_zero():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_A(), 22050,
    samples_to_ms(512, 22050),
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
    align_target="mfcc", aligning="pad", dtw_radius=None,
  )
  assert mcd == 0
  assert pen == 0


def test_removing_silence_from_spec_too_hard_returns_nan_nan():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="spec", silence_threshold_A=100, silence_threshold_B=0,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
    align_target="mfcc", aligning="pad", dtw_radius=None,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="spec", silence_threshold_A=0, silence_threshold_B=100,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
    align_target="mfcc", aligning="pad", dtw_radius=None,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="spec", silence_threshold_A=100, silence_threshold_B=100,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
    align_target="mfcc", aligning="pad", dtw_radius=None,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_removing_silence_from_mel_too_hard_returns_nan_nan():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mel", silence_threshold_A=100, silence_threshold_B=0,
    align_target="mfcc", aligning="pad", dtw_radius=None,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mel", silence_threshold_A=0, silence_threshold_B=100,
    align_target="mfcc", aligning="pad", dtw_radius=None,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mel", silence_threshold_A=100, silence_threshold_B=100,
    align_target="mfcc", aligning="pad", dtw_radius=None,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_removing_silence_from_mfcc_too_hard_returns_nan_nan():
  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mfcc", silence_threshold_A=100, silence_threshold_B=0,
    align_target="mfcc",
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mfcc", silence_threshold_A=0, silence_threshold_B=100,
    align_target="mfcc", aligning="pad", dtw_radius=None,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(
    get_X_km_A(), get_X_km_B(), 22050,
    samples_to_ms(512, 22050),
    remove_silence="mfcc", silence_threshold_A=100, silence_threshold_B=100,
    align_target="mfcc", aligning="pad", dtw_radius=None,
    D=16, s=1, M=20, fmin=0, fmax=22050 // 2,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_unequal_n_fft_raises_error():
  sr2, signalB = wavfile.read(AUDIO_B)
  signalB = norm_audio_signal(signalB)
  X_km_B = get_X_km(signalB, 1024, 1024, 128, "hamming")

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(
      get_X_km_A(), X_km_B, 22050, samples_to_ms(511, 22050))


def test_no_freq_bins_raises_error():
  X_km = np.empty((123, 0), dtype=np.complex128)
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(X_km, X_km, 22050, samples_to_ms(512, 22050))


def test_invalid_radius_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), dtw_radius=0, aligning="dtw")


def test_empty_spec_returns_nan_nan():
  X_km_empty = np.empty((0, get_n_fft_bins(512)), dtype=np.complex128)
  mcd, pen = compare_amplitude_spectrograms(X_km_empty, get_X_km_B(), 22050,
                                            samples_to_ms(512, 22050))
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(get_X_km_A(), X_km_empty, 22050,
                                            samples_to_ms(512, 22050))
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_amplitude_spectrograms(X_km_empty, X_km_empty, 22050,
                                            samples_to_ms(512, 22050))
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_invalid_silence_removal_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="none")


def test_invalid_aligning_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), aligning="none")


def test_invalid_align_target_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), align_target="none")


def test_invalid_sample_rate_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 0,
                                   samples_to_ms(512, 22050))


def test_invalid_n_fft_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050, 0)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050, samples_to_ms(511, 22050))


def test_n_fft_one_larger_raises_no_error():
  compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050, samples_to_ms(513, 22050))


def test_invalid_fmin_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), fmin=-1)


def test_invalid_fmax_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), fmax=0)


def test_invalid_remove_silence_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="none")


def test_invalid_silence_threshold_raises_error():
  # A None
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="sig",
                                   silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="spec",
                                   silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="mel",
                                   silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="mfcc",
                                   silence_threshold_A=None, silence_threshold_B=0.01)
  # B None
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="sig",
                                   silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="spec",
                                   silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="mel",
                                   silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), remove_silence="mfcc",
                                   silence_threshold_A=0.01, silence_threshold_B=None)


def test_removing_silence_after_aligning_raises_error():
  # mel after spec was aligned
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
      remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw", dtw_radius=1,
    )

  # mfcc after spec was aligned
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw", dtw_radius=1,
    )

  # mfcc after mel was aligned
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="mel", aligning="dtw", dtw_radius=1,
    )


def test_D_greater_than_M_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), M=10, D=11)


def test_invalid_D_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), D=0)

  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), D=1)


def test_invalid_M_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), M=0)


def test_invalid_s_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), s=-1)


def test_s_equals_D_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), s=12, D=12)


def test_s_bigger_than_D_raises_error():
  with pytest.raises(ValueError):
    compare_amplitude_spectrograms(get_X_km_A(), get_X_km_B(), 22050,
                                   samples_to_ms(512, 22050), s=13, D=12)


def create_other_outputs():
  targets = []

  # fmax
  for fmax in [None, 8000, 6000, 4000, 2000]:
    targets.append((
      0, fmax, 80, 1, 13, 1
    ))

  # fmin
  for fmin in [0, 1000, 2000, 4000]:
    targets.append((
      fmin, 8000, 80, 1, 13, 1
    ))

  # N
  for n in [20, 40, 60, 80]:
    targets.append((
      0, 8000, n, 1, 13, 1
    ))

  # s, D
  for s, d in [
      (0, 1), (0, 1), (0, 2), (0, 5), (0, 13), (0, 16), (0, 80),
      (1, 2), (1, 5), (1, 13), (1, 16), (1, 80),
      (2, 3), (2, 13), (2, 80),
      (79, 80),
    ]:
    targets.append((
      0, 8000, 80, s, d, 1
    ))

  # dtw_radius
  for r in [1, 2, 3, 5, 10, 20, None]:
    targets.append((
      0, 8000, 80, 1, 13, r
    ))

  # random
  targets.extend([
    (0, 8000, 1, 0, 1, 1),
    (0, 8000, 2, 1, 2, 1),
  ])

  outputs = []

  for fmin, fmax, n, s, d, dtw_radius in targets:
    mcd, pen = compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(),
      22050, samples_to_ms(512, 22050),
      fmin=fmin,
      fmax=fmax,
      M=n,
      s=s,
      D=d,
      align_target="mel",
      aligning="dtw",
      dtw_radius=dtw_radius,
      remove_silence="no",
    )
    outputs.append((fmin, fmax, n, s, d, dtw_radius, mcd, pen))

  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  (TEST_DIR / "test_compare_amplitude_spectrograms_other.pkl").write_bytes(pickle.dumps(outputs))


def test_other_outputs():
  outputs = pickle.loads((TEST_DIR / "test_compare_amplitude_spectrograms_other.pkl").read_bytes())
  for fmin, fmax, n, s, d, dtw_radius, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(), 22050,
      samples_to_ms(512, 22050),
      fmin=fmin,
      fmax=fmax,
      M=n,
      s=s,
      D=d,
      align_target="mel",
      aligning="dtw",
      dtw_radius=dtw_radius,
      remove_silence="no",
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


def create_sil_outputs():
  spec_sil = 0.001
  mel_sil = -7
  mfcc_sil = 0.001

  # pad
  targets = [
      ("no", None, None, "pad", "spec", None),
      ("no", None, None, "pad", "mel", None),
      ("no", None, None, "pad", "mfcc", None),

      ("spec", spec_sil, spec_sil, "pad", "spec", None),
      ("spec", spec_sil, spec_sil, "pad", "mel", None),
      ("spec", spec_sil, spec_sil, "pad", "mfcc", None),

      ("mel", mel_sil, mel_sil, "pad", "mel", None),
      ("mel", mel_sil, mel_sil, "pad", "mfcc", None),

      ("mfcc", mfcc_sil, mfcc_sil, "pad", "mfcc", None),
  ]

  # dtw
  for dtw_radius in [1, 20]:
    targets.extend([
      ("no", None, None, "dtw", "spec", dtw_radius),
      ("no", None, None, "dtw", "mel", dtw_radius),
      ("no", None, None, "dtw", "mfcc", dtw_radius),

      ("spec", spec_sil, spec_sil, "dtw", "spec", dtw_radius),
      ("spec", spec_sil, spec_sil, "dtw", "mel", dtw_radius),
      ("spec", spec_sil, spec_sil, "dtw", "mfcc", dtw_radius),

      ("mel", mel_sil, mel_sil, "dtw", "mel", dtw_radius),
      ("mel", mel_sil, mel_sil, "dtw", "mfcc", dtw_radius),

      ("mfcc", mfcc_sil, mfcc_sil, "dtw", "mfcc", dtw_radius),
    ])

  outputs = []

  for remove_silence, sil_a, sil_b, aligning, target, dtw_radius in targets:
    mcd, pen = compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(),
      22050, samples_to_ms(512, 22050),
      align_target=target,
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
      s=1, D=13, M=80, dtw_radius=dtw_radius,
    )
    outputs.append((remove_silence, sil_a, sil_b, aligning, target, dtw_radius, mcd, pen))
  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  (TEST_DIR / "test_compare_amplitude_spectrograms_sil.pkl").write_bytes(pickle.dumps(outputs))


def test_sil_outputs():
  outputs = pickle.loads(
    (TEST_DIR / "test_compare_amplitude_spectrograms_sil.pkl").read_bytes())
  for remove_silence, sil_a, sil_b, aligning, target, dtw_radius, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_amplitude_spectrograms(
      get_X_km_A(), get_X_km_B(),
      22050, samples_to_ms(512, 22050),
      align_target=target,
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
      s=1, D=13, M=80, dtw_radius=dtw_radius,
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


if __name__ == "__main__":
  create_other_outputs()
  create_sil_outputs()
