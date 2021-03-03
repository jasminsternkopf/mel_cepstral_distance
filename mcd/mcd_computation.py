import librosa
import numpy as np
from fastdtw.fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from mcd.parameters import *


def get_mel_cepstral_distance(input_path: str, output_path: str, N=None):
  input_as_array, _ = librosa.load(input_path, mono=True)
  output_as_array, _ = librosa.load(output_path, mono=True)
  mcd, no_of_frames = mel_cepstral_dist_faster(input_as_array, output_as_array)
  print(
    f"The mel cepstral distance between the given audios is {mcd} (Number of frames: {no_of_frames})")


def get_mfccs(loaded_audio: np.array):
  audio_stft = librosa.stft(loaded_audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, center=False)
  spectogram = np.abs(audio_stft) ** 2
  mel_spectogram = MEL_FILTER_BANK @ spectogram
  log_mel_spectogram = np.log10(mel_spectogram)
  mfccs = COS_MATRIX @ log_mel_spectogram
  return mfccs


def mel_cepstral_dist_faster(input_audio: np.array, output_audio: np.array):
  mfccs_input = get_mfccs(input_audio).T
  mfccs_output = get_mfccs(output_audio).T
  _, path_between_mfccs = fastdtw(mfccs_input, mfccs_output, dist=euclidean)
  path_for_input = list(map(lambda l: l[0], path_between_mfccs))
  path_for_output = list(map(lambda l: l[1], path_between_mfccs))
  mfccs_input = mfccs_input[path_for_input]
  mfccs_output = mfccs_output[path_for_output]
  mfccs_diff = mfccs_output - mfccs_input
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=1)
  mcd = np.mean(mfccs_diff_norms)
  frames = len(mfccs_diff_norms)
  return mcd, frames
