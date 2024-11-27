import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

import numpy as np
import pytest
from scipy.io import wavfile

from mel_cepstral_distance.api import compare_audio_files
from mel_cepstral_distance.helper import samples_to_ms

TEST_DIR = Path("src/mel_cepstral_distance_tests/api_tests")

AUDIO_A = TEST_DIR / "A.wav"
AUDIO_B = TEST_DIR / "B.wav"


def test_aligning_with_pad_returns_same_for_spec_mel_mfcc():
  res = []
  for align_target in ["spec", "mel", "mfcc"]:
    mcd, pen = compare_audio_files(AUDIO_A, AUDIO_B, align_target=align_target, aligning="pad")
    res.append((mcd, pen))
  np.testing.assert_almost_equal(res[0], res[1])
  np.testing.assert_almost_equal(res[0], res[2])
  np.testing.assert_almost_equal(res[1], res[2])


def test_result_changes_after_silence_removal_before_padding_spec():
  mcd, pen = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="spec", aligning="pad",
    remove_silence="sig", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="spec", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_result_changes_after_silence_removal_before_padding_mel():
  mcd, pen = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="mel", aligning="pad",
    remove_silence="spec", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="mel", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_result_changes_after_silence_removal_before_padding_mfcc():
  mcd, pen = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="mfcc", aligning="pad",
    remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
  )
  mcd2, pen2 = compare_audio_files(
    AUDIO_A, AUDIO_B,
    align_target="mfcc", aligning="pad",
    remove_silence="no",
  )

  assert not np.allclose(mcd, mcd2)
  assert not np.allclose(pen, pen2)


def test_same_file_returns_zero():
  mcd, pen = compare_audio_files(AUDIO_A, AUDIO_A)
  assert mcd == 0
  assert pen == 0


def test_invalid_silence_removal_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="none")


def test_invalid_aligning_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, aligning="none")


def test_invalid_align_target_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, align_target="none")


def test_invalid_sample_rate_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, sample_rate=0)


def test_invalid_n_fft_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, n_fft=0)


def test_invalid_win_len_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, win_len=0)


def test_invalid_hop_len_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, hop_len=0)


def test_invalid_window_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, window="none")


def test_invalid_fmin_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, fmin=-1)


def test_invalid_fmax_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, fmax=0)


def test_invalid_remove_silence_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="none")


def test_invalid_silence_threshold_raises_error():
  # A None
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="sil",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="spec",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="mel",
                        silence_threshold_A=None, silence_threshold_B=0.01)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="mfcc",
                        silence_threshold_A=None, silence_threshold_B=0.01)
  # B None
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="sil",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="spec",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="mel",
                        silence_threshold_A=0.01, silence_threshold_B=None)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, remove_silence="mfcc",
                        silence_threshold_A=0.01, silence_threshold_B=None)


def test_removing_silence_after_aligning_raises_error():
  # mel after spec was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      AUDIO_A, AUDIO_B,
      remove_silence="mel", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw",
    )

  # mfcc after spec was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      AUDIO_A, AUDIO_B,
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="spec", aligning="dtw",
    )

  # mfcc after mel was aligned
  with pytest.raises(ValueError):
    compare_audio_files(
      AUDIO_A, AUDIO_B,
      remove_silence="mfcc", silence_threshold_A=0.01, silence_threshold_B=0.01,
      align_target="mel", aligning="dtw",
    )


def test_D_greater_than_N_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, N=10, D=11)


def test_invalid_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, D=0)

  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, D=1)


def test_invalid_N_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, N=0)


def test_invalid_s_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, s=-1)


def test_s_equals_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, s=12, D=12)


def test_s_bigger_than_D_raises_error():
  with pytest.raises(ValueError):
    compare_audio_files(AUDIO_A, AUDIO_B, s=13, D=12)


def create_other_outputs():
  size512 = samples_to_ms(512, 22050)
  size128 = samples_to_ms(128, 22050)

  targets = []

  # sample rate
  for sr in [None, 22050, 16000, 8000, 4000]:
    if sr is None:
      sr_val = 22050
    else:
      sr_val = sr

    targets.append((
      sr,
      samples_to_ms(512, sr_val), samples_to_ms(512, sr_val), samples_to_ms(128, sr_val),
      "hanning", 0, sr_val // 2, 80, 1, 13, True
    ))

  # n_fft, win_len, hop_len
  for n_ffts, win_lens, hop_lens in [
      (512, 512, 256),  # win len == nfft
      (512, 256, 256),  # win len < nfft
      (512, 1024, 256),  # win len > nfft
      (512, 512, 128),
      (1024, 1024, 128),
      (1025, 1025, 129),  # odd
      (1023, 1023, 127),  # odd
    ]:
    targets.append((
      22050, samples_to_ms(n_ffts, 22050), samples_to_ms(win_lens, 22050),
      samples_to_ms(hop_lens, 22050), "hanning", 0, 8000, 80, 1, 13, True
    ))

  # fmax
  for fmax in [None, 8000, 6000, 4000, 2000]:
    targets.append((
      22050, size512, size512, size128, "hanning", 0, fmax, 80, 1, 13, True
    ))

  # fmin
  for fmin in [0, 1000, 2000, 4000]:
    targets.append((
      22050, size512, size512, size128, "hanning", fmin, 8000, 80, 1, 13, True
    ))

  # window
  for window in ["hanning", "hamming"]:
    targets.append((
      22050, size512, size512, size128, window, 0, 8000, 80, 1, 13, True
    ))

  # norm
  for norm in [True, False]:
    targets.append((
      22050, size512, size512, size128, "hanning", 0, 8000, 80, 1, 13, norm
    ))

  # N
  for n in [20, 40, 60, 80]:
    targets.append((
      22050, size512, size512, size128, "hanning", 0, 8000, n, 1, 13, True
    ))

  # s, D
  for s, d in [
      (0, 1), (0, 1), (0, 2), (0, 5), (0, 13), (0, 16), (0, 80),
      (1, 2), (1, 5), (1, 13), (1, 16), (1, 80),
      (2, 3), (2, 13), (2, 80),
      (79, 80),
    ]:
    targets.append((
      22050, size512, size512, size128, "hanning", 0, 8000, 80, s, d, True
    ))

  targets.extend([
    (22050, size512, size512, size128, "hanning", 0, 8000, 1, 0, 1, True),
    (22050, size512, size512, size128, "hanning", 0, 8000, 2, 1, 2, True),
  ])

  outputs = []

  for sample_rate, n_fft, win_len, hop_len, window, fmin, fmax, n, s, d, norm in targets:
    mcd, pen = compare_audio_files(
      AUDIO_A, AUDIO_B,
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
  (TEST_DIR / "test_compare_audio_files_other.pkl").write_bytes(pickle.dumps(outputs))


def test_empty_audio_returns_nan_nan():
  # create empty audio in tmp dir
  with NamedTemporaryFile(suffix=".wav", delete=True, prefix="test_compare_audio_files") as f:
    empty_audio_path = Path(f.name)
    data = np.array([], dtype=np.int16)
    wavfile.write(empty_audio_path, 22050, data)

    mcd, pen = compare_audio_files(AUDIO_A, empty_audio_path, sample_rate=22050, n_fft=512, win_len=512, hop_len=512, window="hanning",
                                   fmin=0, fmax=8000, N=80, s=1, D=13, norm_audio=True, align_target="mel", aligning="dtw", remove_silence="no")
    assert np.isnan(mcd)
    assert np.isnan(pen)

    mcd, pen = compare_audio_files(empty_audio_path, AUDIO_B, sample_rate=22050, n_fft=512, win_len=512, hop_len=512, window="hanning",
                                   fmin=0, fmax=8000, N=80, s=1, D=13, norm_audio=True, align_target="mel", aligning="dtw", remove_silence="no")
    assert np.isnan(mcd)
    assert np.isnan(pen)

    mcd, pen = compare_audio_files(empty_audio_path, empty_audio_path, sample_rate=22050, n_fft=512, win_len=512, hop_len=512, window="hanning",
                                   fmin=0, fmax=8000, N=80, s=1, D=13, norm_audio=True, align_target="mel", aligning="dtw", remove_silence="no")
    assert np.isnan(mcd)
    assert np.isnan(pen)


def test_other_outputs():
  outputs = pickle.loads((TEST_DIR / "test_compare_audio_files_other.pkl").read_bytes())
  for sample_rate, n_fft, win_len, hop_len, window, fmin, fmax, n, s, d, norm, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_audio_files(
      AUDIO_A, AUDIO_B,
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
      AUDIO_A, AUDIO_B,
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
  (TEST_DIR / "test_compare_audio_files_sil.pkl").write_bytes(pickle.dumps(outputs))


def test_sil_outputs():
  outputs = pickle.loads(
    (TEST_DIR / "test_compare_audio_files_sil.pkl").read_bytes())
  for remove_silence, sil_a, sil_b, aligning, target, expected_mcd, expected_pen in outputs:
    mcd, pen = compare_audio_files(
      AUDIO_A, AUDIO_B,
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
