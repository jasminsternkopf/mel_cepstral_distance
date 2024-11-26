import os
from pathlib import Path
from typing import Generator, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.signal import resample


def amp_to_mag(X_km: npt.NDArray[np.complex128]) -> np.ndarray:
  return np.abs(X_km)


def mag_to_energy(X_km: np.ndarray) -> np.ndarray:
  return X_km ** 2


def resample_if_necessary(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
  if sr == target_sr:
    return audio
  target_num_samples = int(len(audio) * target_sr / sr)
  resampled_audio = resample(
    audio, target_num_samples,
    axis=0, window=None, domain='time'
  )
  return resampled_audio


def samples_to_ms(samples: int, sample_rate: int) -> float:
  return samples / sample_rate * 1000


def ms_to_samples(ms: float, sample_rate: int) -> int:
  return int(ms / 1000 * sample_rate)


def remove_silence_rms(audio_signal: np.ndarray, threshold_rms: float, min_silence_samples: int = 256):
  assert 0 <= threshold_rms <= 1
  if threshold_rms == 0:
    return audio_signal
  non_silent_audio = []

  start = 0
  while start < len(audio_signal):
    end = min(start + min_silence_samples, len(audio_signal))
    segment = audio_signal[start:end]

    rms_value = np.sqrt(np.mean(segment**2))

    if rms_value >= threshold_rms:
      non_silent_audio.append(segment)

    start = end

  if non_silent_audio:
    return np.concatenate(non_silent_audio)
  return np.array([], dtype=np.float32)


def get_loudness_vals_X_km(X_km: npt.NDArray[np.complex128]) -> np.ndarray:
  mel_energy = np.mean(amp_to_mag(X_km), axis=1)
  return mel_energy


def detect_non_silence_in_X_km(X_km: npt.NDArray[np.complex128], silence_threshold: float) -> np.ndarray:
  mel_energy = get_loudness_vals_X_km(X_km)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_X_km(X_km: npt.NDArray[np.complex128], silence_threshold: float) -> npt.NDArray[np.complex128]:
  frames = detect_non_silence_in_X_km(X_km, silence_threshold)
  X_km = X_km[frames, :]
  return X_km


def get_loudness_vals_X_kn(X_kn: np.ndarray) -> np.ndarray:
  mel_energy = np.mean(X_kn, axis=1)
  return mel_energy


def detect_non_silence_in_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  mel_energy = get_loudness_vals_X_kn(X_kn)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_X_kn(X_kn, silence_threshold)
  X_kn = X_kn[frames, :]
  return X_kn


def get_loudness_vals_MC_X_ik(MC_X_ik: np.ndarray) -> np.ndarray:
  mel_energy = MC_X_ik[0, :]
  return mel_energy


def detect_non_silence_in_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  # MC_X_ik - Shape: (# MFCC coefficients, time_frames)
  assert len(MC_X_ik) > 0, "Expected non-empty MFCC matrix"
  mel_energy = get_loudness_vals_MC_X_ik(MC_X_ik)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_MC_X_ik(MC_X_ik, silence_threshold)
  MC_X_ik = MC_X_ik[:, frames]
  return MC_X_ik


def norm_audio_signal(audio: np.ndarray) -> np.ndarray:
  audio = audio / np.max(np.abs(audio))
  return audio


def get_hz_points(fmin: float, fmax: float, N: int) -> np.ndarray:
  mel_low = hz_to_mel(fmin)
  mel_high = hz_to_mel(fmax)
  mel_points = np.linspace(mel_low, mel_high, N + 2)
  hz_points = np.array([mel_to_hz(mel_point) for mel_point in mel_points])
  return hz_points


def hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
  assert np.all(hz >= 0), f"Expected positive frequency, but got {hz}"
  return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
  assert np.all(mel >= 0), f"Expected positive mel value, but got {mel}"
  return 700 * (10**(mel / 2595) - 1)


def energy_to_bel(energy: np.ndarray) -> np.ndarray:
  """ 
  Converts energy to bels 
  """
  return 1 * np.log10(energy + np.finfo(float).eps)
