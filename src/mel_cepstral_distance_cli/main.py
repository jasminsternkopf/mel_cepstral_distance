from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable

import numpy as np

from mel_cepstral_distance.mcd_computation import get_metrics_wavs

WINDOWS = [
  "hamming",
  "hann",
  "boxcar",
  "triang",
  "blackman",
  "bartlett",
  "flattop",
  "parzen",
  "bohman",
  "blackmanharris",
  "nuttall",
  "barthann",
  "cosine",
  "exponential",
  "tukey",
  "taylor",
]


def init_mcd_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the Mel-Cepstral Distance and the penalty between two audio files. Both audio files need to have the same sampling rate."
  parser.add_argument("wav_1", type=Path, metavar="WAV1",
                      help="path to the first .wav-file")
  parser.add_argument("wav_2", type=Path, metavar="WAV2",
                      help="path to the second .wav-file")
  parser.add_argument("-f", "--n-fft", type=int, metavar="NFFT", default=1024,
                      help="`NFFT/2+1` is the number of rows of the spectrograms. `NFFT` should be a power of two to optimize the speed of the Fast Fourier Transformation (FFT)")
  parser.add_argument("-l", "--hop-length", type=int, metavar="LENGTH", default=256,
                      help="specifies the number of audio samples between adjacent Short Term Fourier Transformation (STFT)-columns, therefore plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients; for details see `librosa.core.stft`")
  parser.add_argument("-w", "--window", type=str, metavar="NAME", choices=WINDOWS,
                      help="name of the window function; for details see: `scipy.signal.get_window`", default="hamming")
  parser.add_argument("-c", "--center", action="store_true",
                      help="if specified, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2; otherwise `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2")
  parser.add_argument("-m", "--n-mels", type=int, metavar="NMELS", default=20,
                      help="number of Mel bands to generate")
  parser.add_argument("-t", "--htk", action="store_true",
                      help="use HTK formula instead of Slaney when creating the mel-filter bank")
  parser.add_argument("-o", "--norm", type=int, choices=[None, 1], metavar="NORM", default=None,
                      help="determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of the mel band (area normalization); otherwise, leave all the triangles aiming for a peak value of 1.0")
  #parser.add_argument("-y", "--dtype", type=np.dtype, default=np.float64,help="data type of the spectrograms")
  parser.add_argument("-n", "--n-mfcc", type=int, metavar="NMFCC", default=16,
                      help="the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion according to Robert F. Kubichek)")
  parser.add_argument("-d", "--dtw", action="store_true", help="to compute the mel-cepstral distance, the number of frames has to be the same for both audios; if the parameter is specified, Dynamic Time Warping (DTW) is used to align both arrays containing the respective mel-cepstral coefficients, otherwise the array with less columns is filled with zeros from the right side.")
  return print_mcd_dtw_from_paths


def print_mcd_dtw_from_paths(ns: Namespace):
  if not ns.wav_1.is_file():
    raise ValueError("Parameter 'WAV1': File not found!")
  if not ns.wav_2.is_file():
    raise ValueError("Parameter 'WAV2': File not found!")

  mcd, penalty, frames = get_metrics_wavs(
    ns.wav_1,
    ns.wav_2,
    hop_length=ns.hop_length,
    n_fft=ns.n_fft,
    window=ns.window,
    center=ns.center,
    n_mels=ns.n_mels,
    htk=ns.htk,
    norm=ns.norm,
    dtype=np.float64,
    n_mfcc=ns.n_mfcc,
    use_dtw=ns.dtw,
  )

  #print(f"File 1: {ns.wav_1.absolute()}")
  #print(f"File 2: {ns.wav_2.absolute()}")
  print(f"Mel-Cepstral Distance: {mcd}")
  print(f"Penalty: {penalty}")
  print(f"# Frames: {frames}")
