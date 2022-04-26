from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from librosa import load
from librosa.feature import melspectrogram

from mel_cepstral_distance.core import (get_mcd_and_penalty_and_final_frame_number,
                                        get_mfccs_of_mel_spectogram)
from mel_cepstral_distance.types import Frames, MelCepstralDistance, Penalty


def get_metrics_wavs(wav_file_1: Path, wav_file_2: Path, *, hop_length: int = 256, n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True, norm: Optional[Any] = None, dtype: np.dtype = np.float64, n_mfcc: int = 16, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    wav_file_1 : string
        path to the first input WAV file

    wav_file_2 : string
        path to the second input WAV file

    hop_length : int > 0 [scalar]
        specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore
        plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients
        See `librosa.core.stft`

    n_fft     : int > 0 [scalar]
        `n_fft/2+1` is the number of rows of the spectograms. `n_fft` should be a power of two to optimize the speed of
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

    use_dtw:  : bool [scalar]
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
    >>> mcd_12, penalty_12, _ = get_mcd_between_wav_files("exampleaudio_1.wav", "exampleaudio_2.wav")
    >>> mcd_13, penalty_13, _ = get_mcd_between_wav_files("exampleaudio_1.wav". "exampleaudio_3.wav")
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

  mel_spectogram1 = melspectrogram(
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

  mel_spectogram2 = melspectrogram(
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

  return get_metrics_mels(mel_spectogram1, mel_spectogram2, n_mfcc=n_mfcc, take_log=True, use_dtw=use_dtw)


def get_metrics_mels(mel_1: np.ndarray, mel_2: np.ndarray, *, n_mfcc: int = 16, take_log: bool = True, use_dtw: bool = True) -> Tuple[MelCepstralDistance, Penalty, Frames]:
  """Compute the mel-cepstral distance between two audios, a penalty term accounting for the number of frames that has to
  be added to equal both frame numbers or to align the mel-cepstral coefficients if using Dynamic Time Warping and the
  final number of frames that are used to compute the mel-cepstral distance.

    Parameters
    ----------
    mel_1 	  : np.ndarray [shape=(k,n)]
        first mel spectogram

    mel_2     : np.ndarray [shape=(k,m)]
        second mel spectogram

    take_log     : bool
        should be set to `False` if log10 already has been applied to the input mel spectograms, otherwise `True`

    n_mfcc    : int > 0 [scalar]
        the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the
        zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion
        according to Robert F. Kubichek)

    use_dtw:  : bool [scalar]
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
      "The amount of mel-bands that were used to compute the corresponding mel-spectogram have to be equal!")
  mfccs_1 = get_mfccs_of_mel_spectogram(mel_1, n_mfcc, take_log)
  mfccs_2 = get_mfccs_of_mel_spectogram(mel_2, n_mfcc, take_log)
  mcd, penalty, final_frame_number = get_mcd_and_penalty_and_final_frame_number(
    mfccs_1=mfccs_1, mfccs_2=mfccs_2, use_dtw=use_dtw)
  return mcd, penalty, final_frame_number
