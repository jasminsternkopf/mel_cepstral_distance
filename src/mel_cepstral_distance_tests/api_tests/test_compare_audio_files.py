import pickle
from pathlib import Path

import numpy as np
import pytest

from mel_cepstral_distance.api import compare_audio_files
from mel_cepstral_distance.helper import samples_to_ms

A = Path("examples/similar_audios/original.wav")
B = Path("examples/similar_audios/inferred.wav")


def test_aligning_with_pad_returns_same_for_spec_mel_mfcc():
  res = []
  for align_target in ["spec", "mel", "mfcc"]:
    mcd, pen = compare_audio_files(A, B, align_target=align_target, aligning="pad")
    res.append((mcd, pen))
  np.testing.assert_almost_equal(res[0], res[1])
  np.testing.assert_almost_equal(res[0], res[2])
  np.testing.assert_almost_equal(res[1], res[2])


def test_result_changes_after_silence_removal_before_padding_spec():
  mcd, pen = compare_audio_files(
    A, B,
    align_target="spec", aligning="pad",
    remove_silence="sig", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    A, B,
    align_target="spec", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_result_changes_after_silence_removal_before_padding_mel():
  mcd, pen = compare_audio_files(
    A, B,
    align_target="mel", aligning="pad",
    remove_silence="spec", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    A, B,
    align_target="mel", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_result_changes_after_silence_removal_before_padding_mfcc():
  mcd, pen = compare_audio_files(
    A, B,
    align_target="mfcc", aligning="pad",
    remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    A, B,
    align_target="mfcc", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_same_file_returns_zero():
  mcd, pen = compare_audio_files(A, A)
  assert mcd == 0
  assert pen == 0


def test_invalid_silence_removal_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="none")


def test_invalid_aligning_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, aligning="none")


def test_invalid_align_target_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, align_target="none")


def test_invalid_sample_rate_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, sample_rate=0)


def test_invalid_n_fft_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, n_fft=0)


def test_invalid_win_len_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, win_len=0)


def test_invalid_hop_len_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, hop_len=0)


def test_invalid_window_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, window="none")


def test_invalid_fmin_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, fmin=-1)


def test_invalid_fmax_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, fmax=0)


def test_invalid_remove_silence_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="none")


def test_invalid_silence_threshold_raises_error():
  # A None
  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="sil",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="spec",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="mel",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="mfcc",
                        silence_threshold_A=None, silence_threshold_B=0.01)
  # B None
  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="sil",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="spec",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="mel",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, remove_silence="mfcc",
                        silence_threshold_A=0.01, silence_threshold_B=None)


def test_removing_silence_after_aligning_raises_error():
  # mel after spec was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      A, B,
      remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw",
    )

  # mfcc after spec was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      A, B,
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw",
    )

  # mfcc after mel was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      A, B,
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="mel", aligning="dtw",
    )


def test_D_greater_than_N_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, N=10, D=11)


def test_invalid_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, D=0)

  with pytest.raises(ValueError):
    compare_audio_files(A, B, D=1)


def test_invalid_N_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, N=0)


def test_invalid_s_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, s=-1)


def test_s_equals_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, s=12, D=12)


def test_s_bigger_than_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(A, B, s=13, D=12)


def create_other_outputs():
  targets = [
    # sr
    (None, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (16000, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (8000, 32, 32, 16, "hanning", 0, 4000, 80, 1, 13, True),
    (4000, 32, 32, 16, "hanning", 0, 2000, 80, 1, 13, True),

    # n_fft, win_len, hop_len
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 64, 64, 8, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 64, 64, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 64, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 64, 128, 16, "hanning", 0, 8000, 80, 1, 13, True),

    # fmax
    (22050, 32, 32, 16, "hanning", 0, None, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 6000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 4000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 2000, 80, 1, 13, True),

    # fmin
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 1000, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 2000, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 4000, 8000, 80, 1, 13, True),

    # window
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hamming", 0, 8000, 80, 1, 13, True),

    # norm
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, False),

    # N
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 40, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 20, 1, 13, True),

    # s, D
    (22050, 32, 32, 16, "hanning", 0, 8000, 1, 0, 1, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 0, 1, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 0, 5, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 0, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 0, 16, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 0, 80, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 2, 1, 2, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 2, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 13, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 30, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 40, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 16, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 1, 80, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 2, 16, True),
    (22050, 32, 32, 16, "hanning", 0, 8000, 80, 2, 80, True),
  ]

  outputs = []

  for sample_rate, n_fft, win_len, hop_len, window, fmin, fmax, n, s, d, norm in targets:
    mcd, pen = compare_audio_files(
      A, B,
      sample_rate=sample_rate,
      n_fft=n_fft,
      win_len=win_len,
      hop_len=hop_len,
      window=window,
      fmin=fmin,
      fmax=fmax,
      N=n,
      s=s,
      D=d,
      norm_audio=norm,
      align_target="mel",
      aligning="dtw",
      remove_silence="no",
    )
    outputs.append((sample_rate, n_fft, win_len, hop_len,
                   window, fmin, fmax, n, s, d, norm, mcd, pen))

  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  Path("src/mel_cepstral_distance_tests/api_tests/test_compare_audio_files_other.pkl").write_bytes(pickle.dumps(outputs))


def test_other_outputs():
  outputs = pickle.loads(
    Path("src/mel_cepstral_distance_tests/api_tests/test_compare_audio_files_other.pkl").read_bytes())
  for sample_rate, n_fft, win_len, hop_len, window, fmin, fmax, n, s, d, norm, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_audio_files(
      A, B,
      sample_rate=sample_rate,
      n_fft=n_fft,
      win_len=win_len,
      hop_len=hop_len,
      window=window,
      fmin=fmin,
      fmax=fmax,
      N=n,
      s=s,
      D=d,
      norm_audio=norm,
      align_target="mel",
      aligning="dtw",
      remove_silence="no",
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


def create_sil_outputs():
  sig_sil = 0.001
  spec_sil = 0.001
  mel_sil = -7
  mfcc_sil = 0.001

  targets = [
    ("no", None, None, "dtw", "spec"),
    ("no", None, None, "dtw", "mel"),
    ("no", None, None, "dtw", "mfcc"),
    ("no", None, None, "pad", "spec"),
    ("no", None, None, "pad", "mel"),
    ("no", None, None, "pad", "mfcc"),

    ("sig", sig_sil, sig_sil, "dtw", "spec"),
    ("sig", sig_sil, sig_sil, "dtw", "mel"),
    ("sig", sig_sil, sig_sil, "dtw", "mfcc"),
    ("sig", sig_sil, sig_sil, "pad", "spec"),
    ("sig", sig_sil, sig_sil, "pad", "mel"),
    ("sig", sig_sil, sig_sil, "pad", "mfcc"),

    ("spec", spec_sil, spec_sil, "dtw", "spec"),
    ("spec", spec_sil, spec_sil, "dtw", "mel"),
    ("spec", spec_sil, spec_sil, "dtw", "mfcc"),
    ("spec", spec_sil, spec_sil, "pad", "spec"),
    ("spec", spec_sil, spec_sil, "pad", "mel"),
    ("spec", spec_sil, spec_sil, "pad", "mfcc"),

    ("mel", mel_sil, mel_sil, "dtw", "mel"),
    ("mel", mel_sil, mel_sil, "dtw", "mfcc"),
    ("mel", mel_sil, mel_sil, "pad", "mel"),
    ("mel", mel_sil, mel_sil, "pad", "mfcc"),

    ("mfcc", mfcc_sil, mfcc_sil, "dtw", "mfcc"),
    ("mfcc", mfcc_sil, mfcc_sil, "pad", "mfcc"),
  ]

  outputs = []

  for remove_silence, sil_a, sil_b, aligning, target in targets:
    mcd, pen = compare_audio_files(
      A, B,
      sample_rate=22050,
      n_fft=samples_to_ms(512, 22050),
      win_len=samples_to_ms(512, 22050),
      hop_len=samples_to_ms(512 // 4, 22050),
      align_target=target,
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
    )
    outputs.append((remove_silence, sil_a, sil_b, aligning, target, mcd, pen))
  for vals in outputs:
    print("\t".join(str(i) for i in vals))
  Path("src/mel_cepstral_distance_tests/api_tests/test_compare_audio_files_sil.pkl").write_bytes(pickle.dumps(outputs))


def test_sil_outputs():
  outputs = pickle.loads(
    Path("src/mel_cepstral_distance_tests/api_tests/test_compare_audio_files_sil.pkl").read_bytes())
  for remove_silence, sil_a, sil_b, aligning, target, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_audio_files(
      A, B,
      sample_rate=22050,
      n_fft=samples_to_ms(512, 22050),
      win_len=samples_to_ms(512, 22050),
      hop_len=samples_to_ms(512 // 4, 22050),
      align_target=target,
      aligning=aligning,
      remove_silence=remove_silence,
      silence_threshold_A=sil_a,
      silence_threshold_B=sil_b,
    )
    np.testing.assert_almost_equal(mcd, expected_mcd)
    np.testing.assert_almost_equal(pen, expected_pen)


if __name__ == "__main__":
  create_other_outputs()
  create_sil_outputs()
