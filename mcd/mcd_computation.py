from dataclasses import dataclass
from typing import Tuple

import librosa
import numpy as np
from fastdtw.fastdtw import fastdtw
from librosa.feature.spectral import mfcc
from scipy.spatial.distance import euclidean

# @dataclass
# class MCD_Result:
#   mcd: float
#   penalty: float
#   final_frame_number: int
#   added_frames: int


def get_mcd_between_wav_files(wav_file_1: str, wav_file_2: str, hop_length: int = 256, n_fft: int = 1024,
                              window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True,
                              norm=None, dtype=np.float64, n_mfcc: int = 16, use_dtw: bool = True
                              ) -> Tuple[float, float, int]:
  audio_1, sr_1 = librosa.load(wav_file_1, mono=True)
  audio_2, sr_2 = librosa.load(wav_file_2, mono=True)
  return get_mcd_between_audios(
    audio_1=audio_1,
    audio_2=audio_2,
    sr_1=sr_1,
    sr_2=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    n_mfcc=n_mfcc,
    use_dtw=use_dtw
  )


def get_mcd_between_audios(audio_1: np.ndarray, audio_2: np.ndarray, sr_1: int, sr_2: int, hop_length: int = 256,
                           n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20,
                           htk: bool = True, norm=None, dtype=np.float64, n_mfcc: int = 16,
                           use_dtw: bool = True) -> Tuple[float, float, int]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to be added to equal both frame numbers or to align the mel-cepstral coefficients when using DTW and the final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    audio_1 	: np.ndarray [shape=(n,)]
        first audio time-series

    audio_1   : np.ndarray [shape=(m,)]
        second audio time-series

    sr_1        : number > 0 [scalar]
        sampling rate of the first incoming signal

    sr_2        : number > 0 [scalar]
        sampling rate of the second incoming signal

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    n_fft     : int > 0 [scalar]
        number of FFT components
        See `librosa.core.stft`

    window    : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        See `librosa.filters.get_window`

    center    : bool [scalar]
        - If `True`, the signal `audio_i` is padded so that frame
          `D[:, t]` with `D` being the Short-term Fourier transform of the audio is centered at `audio_i[t * hop_length]` for i=1,2
        - If `False`, then `D[:, t]` begins at `audio_i[t * hop_length]` for i=1,2

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    htk       : bool [scalar]
        use HTK formula instead of Slaney when creating the mel-filter bank

    norm      : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band
        (area normalization).  Otherwise, leave all the triangles aiming for
        a peak value of 1.0

    dtype     : np.dtype
        data type of the output

    n_mfcc  : int > 0 [scalar]
        the number of mel-cepstral coefficents that are computed per frame, starting with the first coefficent (the zeroth coefficient is omitted)

    use_dtw:  : bool [scalar]

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """
  if sr_1 != sr_2:
    print("Warning: The sampling rates differ.")
  mfccs_1 = get_mfccs_of_audio(
    audio=audio_1,
    sr=sr_1,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    n_mfcc=n_mfcc
  )
  mfccs_2 = get_mfccs_of_audio(
    audio=audio_2,
    sr=sr_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    n_mfcc=n_mfcc
  )
  return mel_cepstral_distance_and_penalty_and_final_frame_number(mfccs_1, mfccs_2, use_dtw)


def get_mcd_between_mel_spectograms(mel_1: np.ndarray, mel_2: np.ndarray, n_mfcc: int = 16, take_log: bool = True, use_dtw: bool = True) -> Tuple[float, float, int]:
  mfccs_1 = get_mfccs_of_mel_spectogram(mel_1, n_mfcc, take_log)
  mfccs_2 = get_mfccs_of_mel_spectogram(mel_2, n_mfcc, take_log)
  res = mel_cepstral_distance_and_penalty_and_final_frame_number(
    mfccs_1=mfccs_1, mfccs_2=mfccs_2, use_dtw=use_dtw)
  return res


def get_mfccs_of_audio(audio: np.ndarray, sr: int, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming',
                       center: bool = False, n_mels: int = 20, htk: bool = True, norm=None, dtype=np.float64, n_mfcc: int = 16) -> np.ndarray:
  mel_spectogram = librosa.feature.melspectrogram(
    audio, sr=sr, hop_length=hop_length, n_fft=n_fft, window=window, center=center, n_mels=n_mels, htk=htk, norm=norm, dtype=dtype)
  mfccs = get_mfccs_of_mel_spectogram(mel_spectogram, n_mfcc)
  return mfccs


def get_mfccs_of_mel_spectogram(mel_spectogram: np.ndarray, n_mfcc: int, take_log: bool = True):
  mel_spectogram = np.log10(mel_spectogram) if take_log else mel_spectogram
  mfccs = librosa.feature.mfcc(
    S=mel_spectogram,
    n_mfcc=n_mfcc + 1,
    norm=None
  )
  # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth coefficient is omitted
  # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's mfcc uses is 2 times bigger than the one we want to use (which appears in Kubicheks paper)
  mfccs = 1 / 2 * mfccs[1:]
  return mfccs


def mel_cepstral_distance_and_penalty_and_final_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray,
                                                             use_dtw: bool) -> Tuple[float, float, int]:
  former_frame_number_1 = mfccs_1.shape[1]
  former_frame_number_2 = mfccs_2.shape[1]
  mcd, final_frame_number = mel_cepstral_dist_with_equaling_frame_number(
    mfccs_1, mfccs_2, use_dtw)
  penalty = dtw_penalty(former_frame_number_1,
                        former_frame_number_2, final_frame_number)
  return mcd, penalty, final_frame_number


def mel_cepstral_dist_with_equaling_frame_number(mfccs_1: np.ndarray, mfccs_2: np.ndarray,
                                                 use_dtw: bool) -> Tuple[float, int]:
  if mfccs_1.shape[0] != mfccs_2.shape[0]:
    raise Exception("The number of coefficients per frame has to be the same for both inputs.")
  equal_frame_number_mfcc_1, equal_frame_number_mfcc_2 = make_frame_number_equal(
    mfccs_1, mfccs_2, use_dtw)
  return mel_cepstral_dist(equal_frame_number_mfcc_1, equal_frame_number_mfcc_2)


def make_frame_number_equal(mfccs_1: np.ndarray, mfccs_2: np.ndarray, use_dtw: bool) -> Tuple[np.ndarray, np.ndarray]:
  if use_dtw:
    return align_mfccs_with_dtw(mfccs_1.T, mfccs_2.T)
  return fill_rest_with_zeros(mfccs_1, mfccs_2)


def align_mfccs_with_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  _, path_between_mfccs = fastdtw(mfccs_1, mfccs_2, dist=euclidean)
  path_for_input = list(map(lambda l: l[0], path_between_mfccs))
  path_for_output = list(map(lambda l: l[1], path_between_mfccs))
  mfccs_1 = mfccs_1[path_for_input]
  mfccs_2 = mfccs_2[path_for_output]
  return mfccs_1.T, mfccs_2.T


def fill_rest_with_zeros(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  frame_number_1 = mfccs_1.shape[1]
  frame_number_2 = mfccs_2.shape[1]
  diff = abs(frame_number_1 - frame_number_2)
  if diff > 0:
    adding_array = np.zeros(shape=(mfccs_1.shape[0], diff))
    if frame_number_1 < frame_number_2:
      mfccs_1 = np.concatenate((mfccs_1, adding_array), axis=1)
    else:
      mfccs_2 = np.concatenate((mfccs_2, adding_array), axis=1)
  assert mfccs_1.shape == mfccs_2.shape
  return mfccs_1, mfccs_2


def mel_cepstral_dist(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  mfccs_diff = mfccs_1 - mfccs_2
  mfccs_diff_norms = np.linalg.norm(mfccs_diff, axis=0)
  mcd = np.mean(mfccs_diff_norms)
  frame_number = len(mfccs_diff_norms)
  return mcd, frame_number


def dtw_penalty(former_length_1: int, former_length_2: int, length_after_dtw: int) -> float:
  # lies between 0 and 1, the smaller the better
  penalty = 2 - (former_length_1 + former_length_2) / length_after_dtw
  return penalty
