from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.signal import resample


def amp_to_mag(X_km: npt.NDArray[np.complex128]) -> npt.NDArray:
  return np.abs(X_km)


def mag_to_energy(X_km: npt.NDArray) -> npt.NDArray:
  return X_km ** 2


def resample_if_necessary(audio: npt.NDArray, sr: int, target_sr: int) -> npt.NDArray:
  if sr == target_sr:
    return audio
  target_num_samples = int(len(audio) * target_sr / sr)
  resampled_audio: npt.NDArray = resample(
    audio, target_num_samples,
    axis=0, window=None, domain='time'
  )

  return resampled_audio


def samples_to_ms(samples: int, sample_rate: int) -> float:
  return samples / sample_rate * 1000


def ms_to_samples(ms: float, sample_rate: int) -> int:
  return int(ms / 1000 * sample_rate)


def get_n_fft_bins(n_fft: int) -> int:
  return n_fft // 2 + 1


def norm_audio_signal(audio: npt.NDArray) -> npt.NDArray:
  audio = audio / np.max(np.abs(audio))
  return audio


def get_hz_points(fmin: float, fmax: float, M: int) -> npt.NDArray:
  mel_low = hz_to_mel(fmin)
  mel_high = hz_to_mel(fmax)
  mel_points = np.linspace(mel_low, mel_high, M + 2)
  hz_points = np.array([mel_to_hz(mel_point) for mel_point in mel_points])
  return hz_points


def hz_to_mel(hz: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
  assert np.all(hz >= 0), f"Expected positive frequency, but got {hz}"
  return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
  assert np.all(mel >= 0), f"Expected positive mel value, but got {mel}"
  return 700 * (10**(mel / 2595) - 1)


def energy_to_bel(energy: npt.NDArray) -> npt.NDArray:
  """ 
  Converts energy to bels 
  """
  result: npt.NDArray = np.log10(energy + np.finfo(float).eps)
  return result
