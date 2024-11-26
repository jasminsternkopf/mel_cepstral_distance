import os
from pathlib import Path
from typing import Generator, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from fastdtw.fastdtw import fastdtw
from PIL import Image, ImageOps
from scipy.signal import resample

from mel_cepstral_distance.helper import amp_to_mag, energy_to_bel, get_hz_points, mag_to_energy

PLOT_HIGHT = 3


def plot_X_km(X_km: npt.NDArray[np.complex128], sample_rate: int, title: str):
  """
  Plots the energy spectrogram of the given linear amplitude spectrogram.
  """
  n_frames = X_km.shape[0]  # Time frames
  num_freq_bins = X_km.shape[1]  # Frequency bins

  freq_bins = np.linspace(0, sample_rate / 2, num_freq_bins)
  # time_bins = np.arange(num_time_bins) * (hop_len_samples / sample_rate)
  frame_bins = np.arange(n_frames)

  X_km_mag = amp_to_mag(X_km)
  X_km_energy = mag_to_energy(X_km_mag)
  X_km_energy_db = energy_to_bel(X_km_energy) * 10

  # Plotting the spectrogram
  fig, ax = plt.subplots()
  cax = ax.pcolormesh(frame_bins, freq_bins, X_km_energy_db.T, shading='auto')
  fig.colorbar(cax, label='Energy (dB)')
  ax.set_title(f'Linear Spectrogram - {title}')
  ax.set_xlabel('Time [frame]')
  ax.set_ylabel('Frequency [Hz]')

  plt.gcf().set_size_inches(get_plot_width(n_frames), PLOT_HIGHT)
  fig.tight_layout(pad=0.2)

  return fig


def plot_X_kn(X_kn: np.ndarray, fmin: int, fmax: int, title: str):
  """
  Plots the given Mel spectrogram.

  Parameters:
  - X_kn: 2D NumPy array representing the Mel spectrogram (shape: [time_frames, mel_bins]).
  - mel_bank: 2D NumPy array or 1D array representing Mel frequencies (shape: [mel_bins] or [mel_bins, freq_bins]).
  """
  # Extract number of time frames and Mel bins
  n_frames, n_mel_bins = X_kn.shape

  time_axis = np.linspace(0, n_frames + 1, n_frames + 1)
  mel_freqs = get_hz_points(fmin, fmax, n_mel_bins)[1:]

  # X_kn is in Bel
  X_kn_db = X_kn * 10

  # Plotting the Mel spectrogram
  fig, ax = plt.subplots()
  cax = ax.pcolormesh(time_axis, mel_freqs, X_kn_db.T, shading='auto')
  fig.colorbar(cax, label='Energy (dB)')

  # ticks = np.arange(-100, 0 + 20, 20)
  # color_steps = np.arange(-100, 0 + 0.1, 0.1)
  # fig.colorbar(cax, label='Energy [dB]', ax=axes[0], boundaries=color_steps, ticks=ticks)

  ax.set_title(f'Mel Spectrogram - {title}')
  # y achse 1 step
  # plt.yticks(mel_freqs)
  # plt.yscale('log')
  ax.set_yscale('log')
  ticks = np.geomspace(mel_freqs[0], mel_freqs[-1], num=6, dtype=int)

  ax.set_yticks(mel_freqs, minor=True, labels=[])
  ax.minorticks_off()
  ax.set_yticks(
    ticks,
    ticks,
    minor=False,
  )
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


def extract_frames_from_signal(signal: np.ndarray, frames: np.ndarray, hop_len_samples: int):
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


def fill_with_zeros_1d(array_1: np.ndarray, array_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  if array_1.ndim != 1 or array_2.ndim != 1:
    raise ValueError("Both input arrays must be 1-dimensional")

  max_length = max(len(array_1), len(array_2))
  array_1 = np.pad(array_1, (0, max_length - len(array_1)), mode='constant')
  array_2 = np.pad(array_2, (0, max_length - len(array_2)), mode='constant')

  return array_1, array_2


def align_1d_sequences_using_dtw(sequence_1: np.ndarray, sequence_2: np.ndarray):
  _, path = fastdtw(sequence_1, sequence_2)
  path_np = np.array(path)
  stretched_seq_1 = sequence_1[path_np[:, 0]]
  stretched_seq_2 = sequence_2[path_np[:, 1]]
  return stretched_seq_1, stretched_seq_2
