from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional
from typing import OrderedDict as ODType
from typing import Tuple, cast

import numpy as np
import pandas as pd
from librosa import load
from librosa.feature import melspectrogram
from pandas import DataFrame
from tqdm import tqdm

from mel_cepstral_distance.core import (get_mcd_and_penalty_and_final_frame_number,
                                        get_mfccs_of_mel_spectrogram)
from mel_cepstral_distance.helper import get_all_files_in_all_subfolders
from mel_cepstral_distance.types import Frames, MelCepstralDistance, Penalty


def get_metrics_wavs(wav_file_1: Path, wav_file_2: Path, *, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True, norm: Optional[Any] = None, dtype: np.dtype = np.float64, n_mfcc: int = 16, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """
  Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

  Parameters
  ----------
  wav_file_1 : Path
      path to the first input WAV file

  wav_file_2 : Path
      path to the second input WAV file

  hop_length : int > 0 [scalar]
      specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
      plays a role in computing the (mel-)spectrograms which are needed to compute the mel-cepstral coefficients
      See `librosa.core.stft`

  n_fft     : int > 0 [scalar]
      `n_fft/2+1` is the number of rows of the spectrograms. `n_fft` should be a power of two to optimize the speed of
      the Fast Fourier Transformation

  window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a vector or array of length `n_fft`

      See `librosa.filters.get_window`

  center    : bool [scalar]
      - If `True`, the signal `audio_i` is padded so that frame `D[:, t]` with `D` being the Short-term Fourier
        transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
      - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

  n_mels    : int > 0 [scalar]
      number of Mel bands to generate

  htk       : bool [scalar]
      use HTK formula instead of Slaney when creating the mel-filter bank

  norm      : {None, 1, np.inf} [scalar]
      determines if and how the mel weights are normalized: if 1, divide the triangular mel weights by the width of
      the mel band (area normalization).  Otherwise, leave all the triangles aiming for a peak value of 1.0

  dtype     : np.dtype
      data type of the output

  n_mfcc    : int > 0 [scalar]
      the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
      zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
      according to Robert F. Kubichek)

  use_dtw  : bool [scalar]
      to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
      `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
      otherwise the array with less columns is filled with zeros from the right side.

  Returns
  -------
  mcd        : float
      the mel-cepstral distance between the two input audios
  penalty    : float
      a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
      with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
      coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
      each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
      reached if no columns were added to either array.
  final_frame_number : int
      the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
      filling up with zeros

  Example
  --------
  Comparing two audios to another audio using the sum of the mel-cepstral distance and the penalty

  >>> import librosa
  >>> mcd_12, penalty_12, _ = get_metrics_wavs(Path("exampleaudio_1.wav"), Path("exampleaudio_2.wav"))
  >>> mcd_13, penalty_13, _ = get_metrics_wavs(Path("exampleaudio_1.wav"). Path("exampleaudio_3.wav"))
  >>> mcd_with_penalty_12 = mcd_12 + penalty_12
  >>> mcd_with_penalty_13 = mcd_13 + penalty_13
  >>> if mcd_with_penalty_12 < mcd_with_penalty_13:
  >>>   print("Audio 2 seems to be more similar to audio 1 than audio 3.")
  >>> elif mcd_with_penalty_13 < mcd_with_penalty_12:
  >>>   print("Audio 3 seems to be more similar to audio 1 than audio 2.")
  >>> else:
  >>>   print("Audio 2 and audio 3 seem to share the same similarity to audio 1.")
  """

  if not wav_file_1.is_file():
    raise ValueError("Parameter 'wav_file_1': File not found!")
  if not wav_file_2.is_file():
    raise ValueError("Parameter 'wav_file_2': File not found!")

  audio_1, sr_1 = load(wav_file_1, sr=None, mono=True, res_type=None,
                       offset=0.0, duration=None, dtype=np.float32)
  audio_2, sr_2 = load(wav_file_2, sr=None, mono=True, res_type=None,
                       offset=0.0, duration=None, dtype=np.float32)

  if sr_1 != sr_2:
    raise ValueError(
      "Parameters 'wav_file_1' and 'wav_file_2': The sampling rates need to be equal!")

  mel_spectrogram1 = melspectrogram(
    y=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  mel_spectrogram2 = melspectrogram(
    y=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    S=None,
    pad_mode="constant",
    power=2.0,
    win_length=None,
    # librosa.filters.mel arguments:
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    fmin=0.0,
    fmax=None,
  )

  return get_metrics_mels(mel_spectrogram1, mel_spectrogram2, n_mfcc=n_mfcc, take_log=True, use_dtw=use_dtw)


def get_metrics_mels(mel_1: np.ndarray, mel_2: np.ndarray, *, n_mfcc: int = 16, take_log: bool = True, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """
  Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

  Parameters
  ----------
  mel_1 	  : np.ndarray [shape=(k,n)]
      first mel spectrogram

  mel_2     : np.ndarray [shape=(k,m)]
      second mel spectrogram

  take_log     : bool
      should be set to `False` if log10 already has been applied to the input mel spectrograms, otherwise `True`

  n_mfcc    : int > 0 [scalar]
      the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
      zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
      according to Robert F. Kubichek)

  use_dtw  : bool [scalar]
      to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
      `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
      otherwise the array with less columns is filled with zeros from the right side.

  Returns
  -------
  mcd         : float
      the mel-cepstral distance between the two input audios
  penalty     : float
      a term punishing for the number of frames that had to be added to align the mel-cepstral coefficient arrays
      with Dynamic Time Warping (for `use_dtw = True`) or to equal the frame numbers via filling up one mel-cepstral
      coefficient array with zeros (for `use_dtw = False`). The penalty is the sum of the number of added frames of
      each of the two arrays divided by the final frame number (see below). It lies between zero and one, zero is
      reached if no columns were added to either array.
  final_frame_number : int
      the number of columns of one of the mel-cepstral coefficient arrays after applying Dynamic Time Warping or
      filling up with zeros
  """

  if mel_1.shape[0] != mel_2.shape[0]:
    raise ValueError(
      "The amount of mel-bands that were used to compute the corresponding mel-spectrogram have to be equal!")
  mfccs_1 = get_mfccs_of_mel_spectrogram(mel_1, n_mfcc, take_log)
  mfccs_2 = get_mfccs_of_mel_spectrogram(mel_2, n_mfcc, take_log)
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number(
    mfccs_1=mfccs_1, mfccs_2=mfccs_2, use_dtw=use_dtw)
  return mcd, penalty, final_frame_number


def get_metrics_mels_pairwise(folder1: Path, folder2: Path, *, n_mfcc: int = 16, take_log: bool = True, use_dtw: bool = True, silent: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
  """
  Compute the mel-cepstral distance between mel spectrograms (*.npy) pairs that have the same relative path.

  Parameters
  ----------
  folder1 : string
      path to the first folder

  folder2 : string
      path to the second folder

  n_mfcc    : int > 0 [scalar]
      the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
      zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
      according to Robert F. Kubichek)

  take_log     : bool
      should be set to `False` if log10 already has been applied to the input mel spectrograms, otherwise `True`

  use_dtw  : bool [scalar]
      to compute the mel-cepstral distance, the number of frames has to be the same for both audios. If `use_dtw` is
      `True`, Dynamic Time Warping is used to align both arrays containing the respective mel-cepstral coefficients,
      otherwise the array with less columns is filled with zeros from the right side.

  silent     : bool
      should be set to `False` if no progress bar should be displayed

  Returns
  -------
  Values: DataFrame
    Calculate '#Frames', 'MCD', 'PEN' for each file pair including an ALL row containing median values.
  Statistics: DataFrame
    Calculate "Min", "Q1", "Q2", "Q3", "Max", "Mean", "SD", "Kurt" and "Skew" for MCD and PEN.
  failed_paths: List[Path]
    Paths to files that couldn't be processed.
  """
  logger = getLogger(__name__)

  all_files = get_all_files_in_all_subfolders(folder1)
  all_npy_files = sorted(file for file in all_files if file.suffix.lower() == ".npy")

  col_mel1 = "MEL1"
  col_mel2 = "MEL2"
  col_name = "Name"
  col_nmfccs = "#MFCCs"
  col_log = "Take log?"
  col_dtw = "Use DTW?"
  col_bands = "#Mel-bands"
  col_frames1 = "#Frames MEL1"
  col_frames2 = "#Frames MEL2"
  col_frames = "#Frames"
  col_mcd = "MCD"
  col_pen = "PEN"

  col_stats_metric = "Metric"
  col_stats_min = "Min"
  col_stats_q1 = "Q1"
  col_stats_q2 = "Q2"
  col_stats_q3 = "Q3"
  col_stats_max = "Max"
  col_stats_mean = "Mean"
  col_stats_sd = "SD"
  col_stats_kurt = "Kurt"
  col_stats_skew = "Skew"

  results: List[ODType[str, Any]] = []
  errors_on_files = []
  for npy_file_1 in tqdm(all_npy_files, desc="Calculating", unit=" mel(s)", disable=silent):
    npy_file_2: Path = folder2 / npy_file_1.relative_to(folder1)
    if not npy_file_2.is_file():
      logger.warning(
        f"No matching pair for \"{npy_file_1.absolute()}\" at \"{npy_file_2.absolute()}\" found! Skipped.")
      errors_on_files.append(npy_file_1)
      continue

    try:
      mel1: np.ndarray = np.load(npy_file_1)
    except Exception as ex:
      logger.warning(f"File \"{npy_file_1.absolute()}\" couldn't be loaded! Skipped.")
      logger.debug(ex)
      errors_on_files.append(npy_file_1)
      continue

    try:
      mel2: np.ndarray = np.load(npy_file_2)
    except Exception as ex:
      logger.warning(f"File \"{npy_file_2.absolute()}\" couldn't be loaded! Skipped.")
      logger.debug(ex)
      errors_on_files.append(npy_file_2)
      continue

    mcd, penalty, frames = get_metrics_mels(
      mel1,
      mel2,
      n_mfcc=n_mfcc,
      take_log=take_log,
      use_dtw=use_dtw,
    )

    assert mel1.shape[0] == mel2.shape[0]

    results.append(OrderedDict((
      (col_mel1, npy_file_1.absolute()),
      (col_mel2, npy_file_2.absolute()),
      (col_name, npy_file_1.relative_to(folder1).stem),
      (col_nmfccs, n_mfcc),
      (col_log, take_log),
      (col_dtw, use_dtw),
      (col_bands, mel1.shape[0]),
      (col_frames1, mel1.shape[1]),
      (col_frames2, mel2.shape[1]),
      (col_frames, frames),
      (col_mcd, mcd),
      (col_pen, penalty),
    )))

  if len(results) == 0:
    logger.info("No files found!")
    empty_df = DataFrame(data=[], columns=[
      col_mel1,
      col_mel2,
      col_name,
      col_nmfccs,
      col_log,
      col_dtw,
      col_bands,
      col_frames1,
      col_frames2,
      col_frames,
      col_mcd,
      col_pen,
    ])

    empty_stats_df = DataFrame(data=[], columns=[
      col_stats_metric,
      col_stats_min,
      col_stats_q1,
      col_stats_q2,
      col_stats_q3,
      col_stats_max,
      col_stats_mean,
      col_stats_sd,
      col_stats_kurt,
      col_stats_skew,
    ])
    return empty_df, empty_stats_df, errors_on_files

  logger.info(f"Found {len(results)} file pairs.")

  df = DataFrame.from_records(results)

  stats = []
  stats.append(OrderedDict((
    (col_stats_metric, "MCD"),
    (col_stats_min, df[col_mcd].min()),
    (col_stats_q1, df[col_mcd].quantile(0.25)),
    (col_stats_q2, df[col_mcd].quantile(0.50)),
    (col_stats_q3, df[col_mcd].quantile(0.75)),
    (col_stats_max, df[col_mcd].max()),
    (col_stats_mean, df[col_mcd].mean()),
    (col_stats_sd, df[col_mcd].std()),
    (col_stats_kurt, df[col_mcd].kurtosis()),
    (col_stats_skew, df[col_mcd].skew()),
  )))

  stats.append(OrderedDict((
    (col_stats_metric, "PEN"),
    (col_stats_min, df[col_pen].min()),
    (col_stats_q1, df[col_pen].quantile(0.25)),
    (col_stats_q2, df[col_pen].quantile(0.50)),
    (col_stats_q3, df[col_pen].quantile(0.75)),
    (col_stats_max, df[col_pen].max()),
    (col_stats_mean, df[col_pen].mean()),
    (col_stats_sd, df[col_pen].std()),  # σ
    (col_stats_kurt, df[col_pen].kurtosis()),  # K
    (col_stats_skew, df[col_pen].skew()),  # γ
  )))
  stats_df = DataFrame.from_records(stats)

  all_row = {
    col_mel1: cast(Path, folder1).absolute(),
    col_mel2: cast(Path, folder2).absolute(),
    col_name: "ALL",
    col_nmfccs: n_mfcc,
    col_log: take_log,
    col_dtw: use_dtw,
    col_bands: df[col_bands].mean(),
    col_frames1: df[col_frames1].median(),
    col_frames2: df[col_frames2].median(),
    col_frames: df[col_frames].median(),
    col_mcd: df[col_mcd].median(),
    col_pen: df[col_pen].median(),
  }
  df = pd.concat([df, pd.DataFrame.from_records([all_row])], ignore_index=True)

  return df, stats_df, errors_on_files
