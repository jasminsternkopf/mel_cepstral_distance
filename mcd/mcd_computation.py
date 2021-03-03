from argparse import ArgumentParser
from typing import Callable, Tuple

import librosa
import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_mcd_dtw_from_paths(path_1: str, path_2: str, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 20, no_of_coeffs_per_frame: int = 16) -> Tuple[float, int]:
  audio_1, sr_1 = get_audio_from_path(path_1)
  audio_2, sr_2 = get_audio_from_path(path_2)
  spectogram_1 = get_spectogram(audio_1, n_fft, hop_length)
  spectogram_2 = get_spectogram(audio_2, n_fft, hop_length)
  mel_spectogram_1 = get_mel_spectogram(spectogram_1, sr_1, n_mels)
  mel_spectogram_2 = get_mel_spectogram(spectogram_2, sr_2, n_mels)
  return get_mcd_dtw_from_mel_spectograms(mel_spectogram_1, mel_spectogram_2, no_of_coeffs_per_frame)


def get_mcd_dtw_from_mel_spectograms(mel_spectogram_1: np.ndarray, mel_spectogram_2: np.ndarray, no_of_coeffs_per_frame: int = 16) -> Tuple[float, int]:
  mfccs_1 = get_mfccs(mel_spectogram_1, no_of_coeffs_per_frame)
  mfccs_2 = get_mfccs(mel_spectogram_2, no_of_coeffs_per_frame)
  return mel_cepstral_dist_dtw(mfccs_1, mfccs_2)


def get_audio_from_path(path: str) -> Tuple[np.ndarray, int]:
  audio, sr = librosa.load(path, mono=True)
  return audio, sr


def get_spectogram(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
  stft_of_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=False)
  spectogram = np.abs(stft_of_audio) ** 2
  return spectogram


def get_mel_spectogram(spectogram: np.ndarray, sr: int = 22050, n_mels: int = 20) -> np.ndarray:
  n_fft = (spectogram.shape[0] - 1) * 2
  mel_filter_bank = librosa.filters.mel(
    sr=sr, n_fft=n_fft, n_mels=n_mels, norm=None, dtype=np.float64, htk=True)
  mel_spectogram = mel_filter_bank @ spectogram
  return mel_spectogram


def cos_func(i: int, n: int) -> float:
  return np.cos((i + 1) * (n + 1 / 2) * np.pi / 20)


def get_mfccs(mel_spectogram: np.ndarray, no_of_coeffs_per_frame: int = 16) -> np.ndarray:
  log_mel_spectogram = np.log10(mel_spectogram)
  n_mels = mel_spectogram.shape[0]
  cos_matrix = np.fromfunction(cos_func, (no_of_coeffs_per_frame, n_mels), dtype=np.float64)
  mfccs = cos_matrix @ log_mel_spectogram
  return mfccs


def mel_cepstral_dist_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  aligned_mfccs_1, aligned_mfccs_2 = align_mfccs_with_dtw(mfccs_1.T, mfccs_2.T)
  return mel_cepstral_dist(aligned_mfccs_1, aligned_mfccs_2)


def mel_cepstral_dist(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  if mfccs_1.shape[1] != mfccs_2.shape[1]:
    raise Exception(
      "The number of frames has to be the same for both inputs. Please use mel_cepstral_dist_dtw.")
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=1)
  mcd = np.mean(mfccs_diff_norms)
  no_of_frames = len(mfccs_diff_norms)
  return mcd, no_of_frames


def align_mfccs_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  _, path_between_mfccs = fastdtw(mfccs_1, mfccs_2, dist=euclidean)
  path_for_input = list(map(lambda l: l[0], path_between_mfccs))
  path_for_output = list(map(lambda l: l[1], path_between_mfccs))
  mfccs_1 = mfccs_1[path_for_input]
  mfccs_2 = mfccs_2[path_for_output]
  return mfccs_1, mfccs_2
