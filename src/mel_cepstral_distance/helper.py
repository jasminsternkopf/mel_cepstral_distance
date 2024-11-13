import os
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
from scipy.signal import resample


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def fill_with_zeros_2d(array_1: np.ndarray, array_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if array_1.ndim != 2 or array_2.ndim != 2:
    raise ValueError("Both input arrays must be 2-dimensional")

  max_frames = max(array_1.shape[1], array_2.shape[1])
  array_1 = np.pad(array_1, ((0, 0), (0, max_frames - array_1.shape[1])), mode='constant')
  array_2 = np.pad(array_2, ((0, 0), (0, max_frames - array_2.shape[1])), mode='constant')
  return array_1, array_2


def fill_with_zeros_1d(array_1: np.ndarray, array_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if array_1.ndim != 1 or array_2.ndim != 1:
    raise ValueError("Both input arrays must be 1-dimensional")

  max_length = max(len(array_1), len(array_2))
  array_1 = np.pad(array_1, (0, max_length - len(array_1)), mode='constant')
  array_2 = np.pad(array_2, (0, max_length - len(array_2)), mode='constant')

  return array_1, array_2


def resample_if_necessary(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
  if sr == target_sr:
    return audio
  target_num_samples = int(len(audio) * target_sr / sr)
  resampled_audio = resample(
    audio, target_num_samples,
    axis=0, window=None, domain='time'
  )
  return resampled_audio


def resample_signal_numpy(audio, sr, target_sr):
  # Verhältnis der Zielabtastrate zur ursprünglichen Abtastrate berechnen
  ratio = target_sr / sr
  # Neue Anzahl der Samples basierend auf dem Verhältnis berechnen
  target_num_samples = int(len(audio) * ratio)
  # Neue Indexwerte für Interpolation berechnen
  new_indices = np.linspace(0, len(audio) - 1, target_num_samples)
  # Interpolation, um neue Werte für die Zielabtastrate zu berechnen
  resampled_audio = np.interp(new_indices, np.arange(len(audio)), audio)
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


def remove_silence_from_spec(spec: np.ndarray, threshold: float) -> np.ndarray:
  """
  can be mel or normal spec
  """
  # mean or sum
  mel_energy = np.mean(spec, axis=1)
  non_silent_frame_indices = np.where(mel_energy >= threshold)[0]
  mel_spectrogram_trimmed = spec[non_silent_frame_indices, :]
  return mel_spectrogram_trimmed


def norm_audio(audio: np.ndarray) -> np.ndarray:
  audio = audio / np.max(np.abs(audio))
  return audio
