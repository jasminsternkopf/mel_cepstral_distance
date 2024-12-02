import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import numpy as np
import pytest
from scipy.io import wavfile

from mel_cepstral_distance.api import get_amplitude_spectrogram
from mel_cepstral_distance.helper import samples_to_ms

TEST_DIR = Path("src/mel_cepstral_distance_tests/api_tests")

AUDIO_A = TEST_DIR / "A.wav"


def test_uint8_8bitPCM():
  with NamedTemporaryFile(suffix=".wav", delete=True, prefix="test_compare_audio_files") as file_a_tmp:
    audio_a_tmp_path = Path(file_a_tmp.name)

    sr_a, audio_a = wavfile.read(AUDIO_A)
    assert sr_a == 22050
    assert audio_a.dtype == np.int16

    norm_audio = audio_a / 32768.0
    new_audio = ((norm_audio + 1) * 127.5).astype(np.uint8)
    wavfile.write(audio_a_tmp_path, 22050, new_audio)

    res = get_amplitude_spectrogram(
      audio_a_tmp_path,
      sample_rate=22050,
      n_fft=32,
      window="hamming",
      hop_len=16,
      norm_audio=False,
      remove_silence=False,
    )

    assert res.shape == (302, 353)


def test_int16_16bitPCM():
  res = get_amplitude_spectrogram(
    AUDIO_A,
    sample_rate=22050,
    n_fft=32,
    window="hamming",
    hop_len=16,
    norm_audio=False,
    remove_silence=False,
  )

  assert res.shape == (302, 353)


def test_float32_32bitFloat():
  with NamedTemporaryFile(suffix=".wav", delete=True, prefix="test_compare_audio_files") as file_a_tmp:
    audio_a_tmp_path = Path(file_a_tmp.name)

    sr_a, audio_a = wavfile.read(AUDIO_A)
    assert sr_a == 22050
    assert audio_a.dtype == np.int16

    norm_audio = audio_a / 32768.0
    new_audio = norm_audio.astype(np.float32)
    wavfile.write(audio_a_tmp_path, 22050, new_audio)

    res = get_amplitude_spectrogram(
      audio_a_tmp_path,
      sample_rate=22050,
      n_fft=32,
      window="hamming",
      hop_len=16,
      norm_audio=False,
      remove_silence=False,
    )

    assert res.shape == (302, 353)


def test_float64_64bitFloat():
  with NamedTemporaryFile(suffix=".wav", delete=True, prefix="test_compare_audio_files") as file_a_tmp:
    audio_a_tmp_path = Path(file_a_tmp.name)

    sr_a, audio_a = wavfile.read(AUDIO_A)
    assert sr_a == 22050
    assert audio_a.dtype == np.int16

    norm_audio = audio_a / 32768.0
    new_audio = norm_audio.astype(np.float64)
    wavfile.write(audio_a_tmp_path, 22050, new_audio)

    res = get_amplitude_spectrogram(
      audio_a_tmp_path,
      sample_rate=22050,
      n_fft=32,
      window="hamming",
      hop_len=16,
      norm_audio=False,
      remove_silence=False,
    )

    assert res.shape == (302, 353)


def test_sr_resampling():
  res = get_amplitude_spectrogram(
    AUDIO_A,
    sample_rate=44100,
    n_fft=32,
    window="hamming",
    hop_len=16,
    norm_audio=False,
    remove_silence=False,
  )

  assert res.shape == (302, 706)


def test_result_changes_after_silence_removal():
  res = get_amplitude_spectrogram(
    AUDIO_A,
    sample_rate=22050,
    n_fft=32,
    window="hamming",
    hop_len=16,
    norm_audio=True,
    remove_silence=False,
  )

  assert res.shape == (302, 353)

  res = get_amplitude_spectrogram(
    AUDIO_A,
    sample_rate=22050,
    n_fft=32,
    window="hamming",
    hop_len=16,
    norm_audio=True,
    remove_silence=True,
    silence_threshold=0.01,
  )

  assert res.shape == (239, 353)


def test_invalid_sample_rate_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, sample_rate=0)


def test_invalid_n_fft_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, n_fft=0)


def test_invalid_win_len_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, win_len=0)


def test_invalid_hop_len_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, hop_len=0)


def test_invalid_window_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, window="none")


def test_no_silence_threshold_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, remove_silence=True, silence_threshold=None)


def test_removing_silence_from_sig_too_hard_returns_empty():
  res = get_amplitude_spectrogram(
    AUDIO_A,
    sample_rate=22050,
    n_fft=32,
    window="hamming",
    hop_len=16,
    norm_audio=True,
    remove_silence=True,
    silence_threshold=100,
  )

  assert res.shape == (0, 353)


def test_empty_signal_returns_empty():
  with NamedTemporaryFile(suffix=".wav", delete=True, prefix="test_compare_audio_files") as file_a_tmp:
    audio_a_tmp_path = Path(file_a_tmp.name)
    audio_a = np.empty(0, dtype=np.int16)
    wavfile.write(audio_a_tmp_path, 22050, audio_a)

    res = get_amplitude_spectrogram(
      audio_a_tmp_path,
      sample_rate=22050,
      n_fft=32,
      window="hamming",
      hop_len=16,
      norm_audio=False,
      remove_silence=False,
    )

    assert res.shape == (0, 353)


def test_invalid_sig_sil_thres_raises_error():
  with pytest.raises(ValueError):
    get_amplitude_spectrogram(AUDIO_A, remove_silence=True, silence_threshold=-1)


def create_outputs():
  size512 = samples_to_ms(512, 22050)
  size128 = samples_to_ms(128, 22050)

  targets: List[Tuple[int, float, float, float, str, bool, Optional[float]]] = []

  # sample rate
  for sr in [None, 22050, 16000, 8000, 4000]:
    if sr is None:
      sr_val = 22050
    else:
      sr_val = sr

    targets.append((
      sr,
      samples_to_ms(512, sr_val), samples_to_ms(512, sr_val), samples_to_ms(128, sr_val),
      "hanning", True, None
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
      samples_to_ms(hop_lens, 22050), "hanning", True, None
    ))

  # window
  for window in ["hanning", "hamming"]:
    targets.append((
      22050, size512, size512, size128, window, True, None
    ))

  # norm
  for norm in [True, False]:
    targets.append((
      22050, size512, size512, size128, "hanning", norm, None
    ))

  # silence removal
  targets.extend([
    (22050, size512, size512, size128, "hanning", True, None),
    (22050, size512, size512, size128, "hanning", True, 0.01),
    (22050, size512, size512, size128, "hanning", True, 0.1),
  ])

  outputs = []

  for sample_rate, n_fft, win_len, hop_len, window, norm, sil_removal in targets:
    spec = get_amplitude_spectrogram(
      AUDIO_A,
      sample_rate=sample_rate,
      n_fft=n_fft,
      win_len=win_len,
      hop_len=hop_len,
      window=window,
      norm_audio=norm,
      remove_silence=sil_removal is not None,
      silence_threshold=sil_removal,
    )
    outputs.append((sample_rate, n_fft, win_len, hop_len,
                   window, norm, sil_removal, spec))

  for vals in outputs:
    print("\t".join(str(i) if not isinstance(i, np.ndarray) else str(np.mean(i)) for i in vals))
  (TEST_DIR / "test_get_amplitude_spectrogram.pkl").write_bytes(pickle.dumps(outputs))


def test_outputs():
  outputs = pickle.loads((TEST_DIR / "test_get_amplitude_spectrogram.pkl").read_bytes())
  for sample_rate, n_fft, win_len, hop_len, window, norm, sil_removal, expected_spec in outputs:
    spec = get_amplitude_spectrogram(
      AUDIO_A,
      sample_rate=sample_rate,
      n_fft=n_fft,
      win_len=win_len,
      hop_len=hop_len,
      window=window,
      norm_audio=norm,
      remove_silence=sil_removal is not None,
      silence_threshold=sil_removal,
    )
    np.testing.assert_almost_equal(spec, expected_spec)


if __name__ == "__main__":
  create_outputs()
