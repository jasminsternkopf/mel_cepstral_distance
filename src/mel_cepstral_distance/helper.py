import os
from pathlib import Path
from typing import Generator, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample, spectrogram


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


def detect_non_silence_in_mfccs(mfccs: np.ndarray, sil_threshold: float) -> np.ndarray:
  mean = np.mean(mfccs, axis=1)
  non_silent_frame_indices = np.where(mean >= sil_threshold)[0]
  return non_silent_frame_indices


def remove_silence_from_spec(spec: np.ndarray, threshold: float) -> np.ndarray:
  """
  can be mel or normal spec
  """
  # mean or sum
  mel_energy = np.mean(spec, axis=1)
  non_silent_frame_indices = np.where(mel_energy >= threshold)[0]
  mel_spectrogram_trimmed = spec[non_silent_frame_indices, :]
  return mel_spectrogram_trimmed


def detect_non_silence_in_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  # MC_X_ik - Shape: (# MFCC coefficients, time_frames)
  assert len(MC_X_ik) > 0, "Expected non-empty MFCC matrix"
  mel_energy = MC_X_ik[0]
  # mel_energy = np.mean(MC_X_ik, axis=0)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def norm_audio(audio: np.ndarray) -> np.ndarray:
  audio = audio / np.max(np.abs(audio))
  return audio


def get_hz_points(low_freq: float, high_freq: float, N: int) -> np.ndarray:
  mel_low = hz_to_mel(low_freq)
  mel_high = hz_to_mel(high_freq)
  mel_points = np.linspace(mel_low, mel_high, N + 2)
  hz_points = np.array([mel_to_hz(mel_point) for mel_point in mel_points])
  return hz_points


def hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
  assert np.all(hz >= 0), f"Expected positive frequency, but got {hz}"
  return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
  assert np.all(mel >= 0), f"Expected positive mel value, but got {mel}"
  return 700 * (10**(mel / 2595) - 1)


def plot_X_km(X_km: np.ndarray, sample_rate: int, hop_len_samples: int):
  # Calculate time and frequency axes
  num_time_bins = X_km.shape[0]  # Time frames
  num_freq_bins = X_km.shape[1]  # Frequency bins

  # Calculate frequency and time bins
  freq_bins = np.linspace(0, sample_rate / 2, num_freq_bins)
  time_bins = np.arange(num_time_bins) * (hop_len_samples / sample_rate)

  # Calculate magnitude spectrogram
  magnitude_spectrogram = np.abs(X_km)

  # Plotting the spectrogram
  fig = plt.figure(figsize=(num_time_bins / 50, num_freq_bins / 100))
  plt.pcolormesh(time_bins, freq_bins, 20 *
                 np.log10(magnitude_spectrogram + 1e-10).T, shading='auto')
  plt.title('STFT Magnitude Spectrogram')
  plt.xlabel('Time [s]')
  plt.ylabel('Frequency [Hz]')
  plt.colorbar(label='Magnitude (dB)')
  return fig


def plot_X_kn(X_kn: np.ndarray, low_freq: int = 0, high_freq: int = 8000):
  """
  Plots the given Mel spectrogram.

  Parameters:
  - X_kn: 2D NumPy array representing the Mel spectrogram (shape: [time_frames, mel_bins]).
  - mel_bank: 2D NumPy array or 1D array representing Mel frequencies (shape: [mel_bins] or [mel_bins, freq_bins]).
  """
  # Extract number of time frames and Mel bins
  n_time_frames, n_mel_bins = X_kn.shape

  time_axis = np.linspace(0, n_time_frames + 1, n_time_frames + 1)
  mel_freqs = get_hz_points(low_freq, high_freq, n_mel_bins)[:-1]
  # mel_freqs = np.arange(n_mel_bins + 1)

  # Plotting the Mel spectrogram
  fig = plt.figure(figsize=(n_time_frames / 50, n_mel_bins / 6))
  plt.pcolormesh(time_axis, mel_freqs, X_kn.T, shading='auto')

  plt.title('Mel Spectrogram')
  # y achse 1 step
  # plt.yticks(mel_freqs)
  # plt.yscale('log')
  plt.xlabel('Time Frames')
  plt.ylabel('Mel Frequency Bins')
  plt.colorbar(label='Power (dB)')

  plt.tight_layout(
    pad=0.2,
  )
  plt.show()
  return fig


def plot_MC_X_ik(MC_X_ik: np.ndarray, title: str = "MFCC Heatmap"):
  """
  Plots the MFCC matrix as a heatmap.

  Parameters:
  - MC_X_ik: 2D NumPy array representing MFCC coefficients (shape: [mfcc_coefficients, time_frames]).
  - title: Title of the plot (default is "MFCC Heatmap").
  """
  n_mfcc_coeff, n_time_frames = MC_X_ik.shape

  # Time and MFCC coefficient axes
  time_axis = np.arange(n_time_frames)
  mfcc_axis = np.arange(1, n_mfcc_coeff + 1)  # Typically 1-based index for MFCC coefficients
  # Typically 1-based index for MFCC coefficients
  mfcc_axis_label = np.arange(2, n_mfcc_coeff + 1, 2)

  # Plotting the MFCC heatmap
  # fig, ax = plt.subplots(figsize=(10, 6))
  fig, ax = plt.subplots(figsize=(n_time_frames / 50, n_mfcc_coeff / 6))
  cax = ax.pcolormesh(time_axis, mfcc_axis, MC_X_ik, shading='auto')
  fig.colorbar(cax, label='MFCC Coefficient Value')

  plt.title(title)
  plt.xlabel('Time Frames')
  plt.ylabel('MFCC Coefficients')
  # plt.yticks(mfcc_axis)
  plt.yticks(mfcc_axis_label)
  plt.tight_layout()
  # plt.show()

  return fig
