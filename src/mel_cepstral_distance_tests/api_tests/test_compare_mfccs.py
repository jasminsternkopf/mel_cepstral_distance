import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from mel_cepstral_distance.api import compare_mfccs
from mel_cepstral_distance.computation import get_MC_X_ik, get_w_n_m, get_X_km, get_X_kn
from mel_cepstral_distance.helper import norm_audio_signal

TEST_DIR = Path("src/mel_cepstral_distance_tests/api_tests")

AUDIO_A = TEST_DIR / "A.wav"
AUDIO_B = TEST_DIR / "B.wav"

M = 80


def test_zero_dim_return_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((0, M)), get_MC_Y_ik_B())

  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), np.zeros((0, M)))

  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((0, M)), np.zeros((0, M)))


def test_one_dim_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((302)), get_MC_Y_ik_B())

  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), np.zeros((302)))

  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((302)), np.zeros((302)))


def test_three_dim_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((302, M, 3)), get_MC_Y_ik_B())

  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), np.zeros((302, M, 3)))

  with pytest.raises(ValueError):
    compare_mfccs(np.zeros((302, M, 3)), np.zeros((302, M, 3)))


def get_MC_X_ik_A():
  sr1, signalA = wavfile.read(AUDIO_A)
  signalA = norm_audio_signal(signalA)
  X_km = get_X_km(signalA, 512, 512, 128, "hamming")
  w_n_m = get_w_n_m(sr1, 512, M, 0, sr1 // 2)
  X_kn = get_X_kn(X_km, w_n_m)
  MC_X_ik = get_MC_X_ik(X_kn, M)
  return MC_X_ik


def get_MC_Y_ik_B():
  sr2, signalB = wavfile.read(AUDIO_B)
  signalB = norm_audio_signal(signalB)
  X_km = get_X_km(signalB, 512, 512, 128, "hamming")
  w_n_m = get_w_n_m(sr2, 512, M, 0, sr2 // 2)
  X_kn = get_X_kn(X_km, w_n_m)
  MC_Y_ik = get_MC_X_ik(X_kn, M)

  return MC_Y_ik


def test_same_mfccs_returns_zero():
  mcd, pen = compare_mfccs(
    get_MC_X_ik_A(), get_MC_X_ik_A(),
    D=16, s=1, aligning="pad", remove_silence=False,
  )
  assert mcd == 0
  assert pen == 0


def test_removing_silence_too_hard_returns_nan_nan():
  mcd, pen = compare_mfccs(
    get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence=True, silence_threshold_A=100, silence_threshold_B=0,
    aligning="pad", D=16, s=1,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_mfccs(
    get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence=True,
    silence_threshold_A=0, silence_threshold_B=100,
    aligning="pad", D=16, s=1,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_mfccs(
    get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence=True,
    silence_threshold_A=100, silence_threshold_B=100,
    aligning="pad", D=16, s=1,
  )
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_empty_returns_nan_nan():
  MC_empty = np.empty((M, 0))
  mcd, pen = compare_mfccs(MC_empty, get_MC_Y_ik_B())
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_mfccs(get_MC_X_ik_A(), MC_empty)
  assert np.isnan(mcd)
  assert np.isnan(pen)

  mcd, pen = compare_mfccs(MC_empty, MC_empty)
  assert np.isnan(mcd)
  assert np.isnan(pen)


def test_invalid_aligning_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), aligning="none")


def test_invalid_remove_silence_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence="none")


def test_invalid_silence_threshold_raises_error():
  # A None
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence=True,
                  silence_threshold_A=None, silence_threshold_B=0.01)

  # B None
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), remove_silence=True,
                  silence_threshold_A=0.01, silence_threshold_B=None)


def test_D_greater_than_M_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), D=M + 1)


def test_invalid_D_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), D=0)

  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), D=1)


def test_invalid_s_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), s=-1)


def test_s_equals_D_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), s=12, D=12)


def test_s_bigger_than_D_raises_error():
  with pytest.raises(ValueError):
    compare_mfccs(get_MC_X_ik_A(), get_MC_Y_ik_B(), s=13, D=12)


def create_other_outputs():
  targets = []

  # s, D
  for s, d in [
      (0, 1), (0, 1), (0, 2), (0, 5), (0, 13), (0, 16), (0, 80),
      (1, 2), (1, 5), (1, 13), (1, 16), (1, 80),
      (2, 3), (2, 13), (2, 80),
      (79, 80),
    ]:
    targets.append((
      s, d
    ))

  outputs = []

  for s, d in targets:
    mcd, pen = compare_mfccs(
      get_MC_X_ik_A(), get_MC_Y_ik_B(),
      s=s,
      D=d,
      aligning="dtw",
      remove_silence=False,
    )
    outputs.append((s, d, mcd, pen))

  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  (TEST_DIR / "test_compare_mfccs_other.pkl").write_bytes(pickle.dumps(outputs))


def test_other_outputs():
  outputs = pickle.loads((TEST_DIR / "test_compare_mfccs_other.pkl").read_bytes())
  for s, d, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_mfccs(
      get_MC_X_ik_A(), get_MC_Y_ik_B(),
      s=s,
      D=d,
      aligning="dtw",
      remove_silence=False,
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


def create_sil_outputs():
  mfcc_sil = 0.001

  targets = [
    (False, None, None, "dtw"),
    (False, None, None, "pad"),

    (True, mfcc_sil, mfcc_sil, "dtw"),
    (True, mfcc_sil, mfcc_sil, "pad"),
  ]

  outputs = []

  for remove_silence, sil_a, sil_b, aligning in targets:
    mcd, pen = compare_mfccs(
      get_MC_X_ik_A(), get_MC_Y_ik_B(),
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
      s=1, D=13,
    )
    outputs.append((remove_silence, sil_a, sil_b, aligning, mcd, pen))
  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  (TEST_DIR / "test_compare_mfccs_sil.pkl").write_bytes(pickle.dumps(outputs))


def test_sil_outputs():
  outputs = pickle.loads(
    (TEST_DIR / "test_compare_mfccs_sil.pkl").read_bytes())
  for remove_silence, sil_a, sil_b, aligning, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_mfccs(
      get_MC_X_ik_A(), get_MC_Y_ik_B(),
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
      s=1, D=13,
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


if __name__ == "__main__":
  create_other_outputs()
  create_sil_outputs()
