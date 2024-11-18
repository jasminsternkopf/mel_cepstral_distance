import os
from pathlib import Path
from typing import Generator, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import resample

PLOT_HIGHT = 3


def get_penalty(former_length_1: int, former_length_2: int, length_after_equaling: int) -> float:
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_equaling
  return penalty


def get_all_files_in_all_subfolders(directory: Path) -> Generator[Path, None, None]:
  for root, _, files in os.walk(directory):
    for name in files:
      file_path = Path(root) / name
      yield file_path


def fill_with_zeros_2d(array_1: np.ndarray, array_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if array_1.ndim != 2 or array_2.ndim != 2:
    raise ValueError("Both input arrays must be 2-dimensional")

  if array_1.shape[1] == array_2.shape[1]:
    return array_1, array_2

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


def extract_extract_frames_from_signal(signal: np.ndarray, frames: np.ndarray, hop_len_samples: int):
  """
  Extrahiert die nicht-stillen Teile eines Audiosignals basierend auf gegebenen Frame-Indizes.

  Parameters:
  signal (numpy array): Das Eingabe-Audiosignal.
  non_silent_frames (numpy array): Indizes der nicht-stillen Frames.
  hop_len_samples (int): Die Hop-Größe in Samples.

  Returns:
  numpy array: Das modifizierte Audiosignal, das nur die nicht-stillen Frames enthält.
  """
  extracted_signal = []

  for frame_idx in frames:
    start = int(frame_idx * hop_len_samples)
    end = int((frame_idx + 1) * hop_len_samples)

    end = min(end, len(signal))
    extracted_signal.extend(signal[start:end])

  return np.array(extracted_signal)


def remove_silence_from_spec(spec: np.ndarray, threshold: float) -> np.ndarray:
  """
  can be mel or normal spec
  """
  # mean or sum
  mel_energy = np.mean(spec, axis=1)
  non_silent_frame_indices = np.where(mel_energy >= threshold)[0]
  mel_spectrogram_trimmed = spec[non_silent_frame_indices, :]
  return mel_spectrogram_trimmed


def remove_silence_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_X_kn(X_kn, silence_threshold)
  X_kn = X_kn[frames, :]
  return X_kn


def remove_silence_X_km(X_km: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_X_km(X_km, silence_threshold)
  X_km = X_km[frames, :]
  return X_km


def get_silence_vals_X_kn(X_kn: np.ndarray) -> np.ndarray:
  mel_energy = np.mean(X_kn, axis=1)
  return mel_energy


def detect_non_silence_in_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  mel_energy = get_silence_vals_X_kn(X_kn)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def get_silence_vals_X_km(X_km: np.ndarray) -> np.ndarray:
  mel_energy = np.mean(X_km, axis=1)
  return mel_energy


def detect_non_silence_in_X_km(X_km: np.ndarray, silence_threshold: float) -> np.ndarray:
  mel_energy = get_silence_vals_X_km(X_km)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_MC_X_ik(MC_X_ik, silence_threshold)
  MC_X_ik = MC_X_ik[:, frames]
  return MC_X_ik


def get_silence_vals_MC_X_ik(MC_X_ik: np.ndarray) -> np.ndarray:
  mel_energy = MC_X_ik[0, :]
  return mel_energy


def detect_non_silence_in_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  # MC_X_ik - Shape: (# MFCC coefficients, time_frames)
  assert len(MC_X_ik) > 0, "Expected non-empty MFCC matrix"
  mel_energy = get_silence_vals_MC_X_ik(MC_X_ik)
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


def plot_X_km(X_km: np.ndarray, sample_rate: int, title: str):
  # Calculate time and frequency axes
  n_frames = X_km.shape[0]  # Time frames
  num_freq_bins = X_km.shape[1]  # Frequency bins

  # Calculate frequency and time bins
  freq_bins = np.linspace(0, sample_rate / 2, num_freq_bins)
  # time_bins = np.arange(num_time_bins) * (hop_len_samples / sample_rate)
  frame_bins = np.arange(n_frames)

  # Calculate magnitude spectrogram
  magnitude_spectrogram = np.abs(X_km)

  # Plotting the spectrogram
  fig, ax = plt.subplots()
  cax = ax.pcolormesh(frame_bins, freq_bins, 20 *
                      np.log10(magnitude_spectrogram + 1e-10).T, shading='auto')
  fig.colorbar(cax, label='Magnitude (dB)')
  ax.set_title(f'STFT Magnitude Spectrogram - {title}')
  ax.set_xlabel('Time [frame]')
  ax.set_ylabel('Frequency [Hz]')

  plt.gcf().set_size_inches(get_plot_width(n_frames), PLOT_HIGHT)
  fig.tight_layout(pad=0.2)

  return fig


def plot_X_kn(X_kn: np.ndarray, low_freq: int, high_freq: int, title: str):
  """
  Plots the given Mel spectrogram.

  Parameters:
  - X_kn: 2D NumPy array representing the Mel spectrogram (shape: [time_frames, mel_bins]).
  - mel_bank: 2D NumPy array or 1D array representing Mel frequencies (shape: [mel_bins] or [mel_bins, freq_bins]).
  """
  # Extract number of time frames and Mel bins
  n_frames, n_mel_bins = X_kn.shape

  time_axis = np.linspace(0, n_frames + 1, n_frames + 1)
  mel_freqs = get_hz_points(low_freq, high_freq, n_mel_bins)[:-1]

  # Plotting the Mel spectrogram
  fig, ax = plt.subplots()
  cax = ax.pcolormesh(time_axis, mel_freqs, X_kn.T, shading='auto')
  fig.colorbar(cax, label='Power (dB)')

  ax.set_title(f'Mel Spectrogram - {title}')
  # y achse 1 step
  # plt.yticks(mel_freqs)
  # plt.yscale('log')
  ax.set_xlabel('Time [frame]')
  ax.set_ylabel('Frequency [Hz]')

  plt.gcf().set_size_inches(get_plot_width(n_frames), PLOT_HIGHT)
  fig.tight_layout(pad=0.2)

  return fig


def plot_MC_X_ik(MC_X_ik: np.ndarray, title: str):
  """
  Plots the MFCC matrix as a heatmap.

  Parameters:
  - MC_X_ik: 2D NumPy array representing MFCC coefficients (shape: [mfcc_coefficients, time_frames]).
  - title: Title of the plot (default is "MFCC Heatmap").
  """
  n_mfcc_coeff, n_frames = MC_X_ik.shape

  # Time and MFCC coefficient axes
  time_axis = np.arange(n_frames)
  mfcc_axis = np.arange(1, n_mfcc_coeff + 1)  # Typically 1-based index for MFCC coefficients
  # Typically 1-based index for MFCC coefficients
  mfcc_axis_label = np.arange(2, n_mfcc_coeff + 1, 2)

  # Plotting the MFCC heatmap
  # fig, ax = plt.subplots(figsize=(10, 6))
  fig, ax = plt.subplots()
  cax = ax.pcolormesh(time_axis, mfcc_axis, MC_X_ik, shading='auto')
  fig.colorbar(cax, label='Value')

  ax.set_title(f"MFCC Heatmap - {title}")
  ax.set_xlabel('Time [frame]')
  ax.set_ylabel('MFCC Coefficient')
  ax.set_yticks(mfcc_axis_label)

  plt.gcf().set_size_inches(get_plot_width(n_frames), PLOT_HIGHT)
  fig.tight_layout(pad=0.2)

  return fig


def stack_images_vertically(image_paths, output_path):
  """
  Stacks images vertically from a list of image paths and saves the result.

  Args:
      image_paths (list): List of paths to the input images.
      output_path (str): Path to save the output stacked image.

  Returns:
      None
  """
  if not image_paths:
    raise ValueError("The image_paths list cannot be empty.")

  # Load all images
  images = [Image.open(path) for path in image_paths]

  from PIL import ImageOps

  # Load all images and add a black border to each
  images = [ImageOps.expand(img, border=10, fill='black') for img in images]

  # Get the width and total height of the stacked image
  max_width = max(image.width for image in images)
  total_height = sum(image.height for image in images)

  # Create a new blank image with the calculated dimensions
  stacked_image = Image.new("RGB", (max_width, total_height), color="red")

  # Paste each image into the stacked image
  y_offset = 0
  for image in images:
    stacked_image.paste(image, (0, y_offset))
    y_offset += image.height

  # Save the stacked image
  stacked_image.save(output_path)


def get_plot_width(n_time_frames: int) -> float:
  return n_time_frames / 50
