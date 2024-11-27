from logging import getLogger
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from mel_cepstral_distance.alignment import align_MC, align_X_km, align_X_kn
from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_w_n_m,
                                               get_X_km, get_X_kn)
from mel_cepstral_distance.helper import (get_n_fft_bins, ms_to_samples, norm_audio_signal,
                                          resample_if_necessary)
from mel_cepstral_distance.silence import (remove_silence_MC_X_ik, remove_silence_rms,
                                           remove_silence_X_km, remove_silence_X_kn)


def compare_audio_files(audio_A: Path, audio_B: Path, *, sample_rate: Optional[int] = None, n_fft: float = 32, win_len: float = 32, hop_len: float = 16, window: Literal["hamming", "hanning"] = "hanning", fmin: int = 0, fmax: Optional[int] = None, N: int = 20, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["spec", "mel", "mfcc"] = "mel", remove_silence: Literal["no", "sig", "spec", "mel", "mfcc"] = "no", silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None, norm_audio: bool = True) -> Tuple[float, float]:
  """
  - silence is removed before alignment
  - high freq is max sr/2
  - n_fft should be equal to win_len
  - n_fft should be a power of 2 in samples
  """
  if remove_silence not in ["no", "sig", "spec", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'sig', 'spec', 'mel' or 'mfcc'")

  if sample_rate is not None and not 0 < sample_rate:
    raise ValueError("sample_rate must be > 0")

  if not n_fft > 0:
    raise ValueError("n_fft must be > 0")

  if not 0 < win_len:
    raise ValueError("win_len must be > 0")

  if not 0 < hop_len:
    raise ValueError("hop_len must be > 0")

  if window not in ["hamming", "hanning"]:
    raise ValueError("window must be 'hamming' or 'hanning'")

  sr1, signalA = wavfile.read(audio_A)
  sr2, signalB = wavfile.read(audio_B)

  # convert to float betweet 0 and 1

  if len(signalA) == 0:
    logger = getLogger(__name__)
    logger.warning("audio A is empty")
    return np.nan, np.nan

  if len(signalB) == 0:
    logger = getLogger(__name__)
    logger.warning("audio B is empty")
    return np.nan, np.nan

  if sample_rate is None:
    sample_rate = min(sr1, sr2)

  signalA = resample_if_necessary(signalA, sr1, sample_rate)
  signalB = resample_if_necessary(signalB, sr2, sample_rate)

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  n_fft_is_two_power = n_fft_samples & (n_fft_samples - 1) == 0

  if not n_fft_is_two_power:
    logger = getLogger(__name__)
    logger.warning(
      f"n_fft ({n_fft}ms / {n_fft_samples} samples) should be a power of 2 in samples for faster computation")

  if n_fft != win_len:
    logger = getLogger(__name__)
    logger.warning(f"n_fft ({n_fft}ms) should be equal to win_len ({win_len}ms)")
    if n_fft < win_len:
      logger.warning(f"truncating windows to n_fft ({n_fft}ms)")
    else:
      assert n_fft > win_len
      logger.warning(f"padding windows to n_fft ({n_fft}ms)")

  if norm_audio:
    signalA = norm_audio_signal(signalA)
    signalB = norm_audio_signal(signalB)

  win_len_samples = ms_to_samples(win_len, sample_rate)

  if remove_silence == "sig":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")

    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    if not 0 <= silence_threshold_A:
      raise ValueError("silence_threshold_A must be greater than or equal to 0 RMS")

    if not 0 <= silence_threshold_B:
      raise ValueError("silence_threshold_B must be greater than or equal to 0 RMS")

    signalA = remove_silence_rms(
      signalA, silence_threshold_A,
      min_silence_samples=win_len_samples
    )

    signalB = remove_silence_rms(
      signalB, silence_threshold_B,
      min_silence_samples=win_len_samples
    )

    if len(signalA) == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, audio A is empty")
      return np.nan, np.nan

    if len(signalB) == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, audio B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  # STFT - Shape: (#Frames, Bins)
  hop_len_samples = ms_to_samples(hop_len, sample_rate)
  X_km_A = get_X_km(signalA, n_fft_samples, win_len_samples, hop_len_samples, window)
  X_km_B = get_X_km(signalB, n_fft_samples, win_len_samples, hop_len_samples, window)

  mean_mcd_over_all_k, res_penalty = compare_amplitude_spectrograms(
    X_km_A, X_km_B, sample_rate, n_fft, fmin=fmin, fmax=fmax, N=N, s=s, D=D, aligning=aligning, align_target=align_target, remove_silence=remove_silence, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B
  )

  return mean_mcd_over_all_k, res_penalty


def compare_amplitude_spectrograms(X_km_A: npt.NDArray[np.complex128], X_km_B: npt.NDArray[np.complex128], sample_rate: int, n_fft: float, *, fmin: int = 0, fmax: Optional[int] = None, N: int = 20, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["spec", "mel", "mfcc"] = "spec", remove_silence: Literal["no", "spec", "mel", "mfcc"] = "no", silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None) -> Tuple[float, float]:
  if X_km_A.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("spectrogram A is empty")
    return np.nan, np.nan

  if X_km_B.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("spectrogram B is empty")
    return np.nan, np.nan

  if not X_km_A.shape[1] == X_km_B.shape[1]:
    raise ValueError(
      f"both spectrograms must have the same number of frequency bins but got {X_km_A.shape[1]} != {X_km_B.shape[1]}")

  assert X_km_A.shape[1] == X_km_B.shape[1]
  n_fft_bins = X_km_A.shape[1]

  if n_fft_bins == 0:
    raise ValueError("spectrograms have no frequency bins")

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  if get_n_fft_bins(n_fft_samples) != n_fft_bins:
    raise ValueError(
      f"n_fft (in samples) // 2 + 1 must match the number of frequency bins in the spectrogram but got {n_fft_samples // 2 + 1} != {n_fft_bins}")
  assert n_fft > 0
  assert sample_rate > 0

  if aligning not in ["pad", "dtw"]:
    raise ValueError("aligning must be 'pad' or 'dtw'")

  if remove_silence not in ["no", "spec", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'spec', 'mel' or 'mfcc'")

  if align_target == "spec":
    if remove_silence == "mel":
      raise ValueError(
        "cannot remove silence from mel-spectrogram after both spectrograms were aligned")
    if remove_silence == "mfcc":
      raise ValueError(
        "cannot remove silence from MFCCs after both spectrograms were aligned")

  if fmax is not None:
    if not 0 < fmax <= sample_rate // 2:
      raise ValueError(f"fmax must be in (0, sample_rate // 2], i.e., (0, {sample_rate//2}]")
  else:
    fmax = sample_rate // 2

  if not 0 <= fmin < fmax:
    raise ValueError(f"fmin must be in [0, fmax), i.e., [0, {fmax})")

  if not N > 0:
    raise ValueError("N must be > 0")

  if remove_silence == "spec":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    X_km_A = remove_silence_X_km(X_km_A, silence_threshold_A)
    X_km_B = remove_silence_X_km(X_km_B, silence_threshold_B)

    if X_km_A.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, spectrogram A is empty")
      return np.nan, np.nan

    if X_km_B.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, spectrogram B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  penalty: float = None
  aligned_here: bool = False
  if align_target == "spec":
    X_km_A, X_km_B, penalty = align_X_km(X_km_A, X_km_B, aligning)
    aligned_here = True
    align_target = "mel"
    aligning = "pad"

  # Mel-Bank - Shape: (N, #Frames)
  w_n_m = get_w_n_m(sample_rate, n_fft_samples, N, fmin, fmax)

  # Mel-Spectrogram - Shape: (#Frames, #N)
  X_kn_A = get_X_kn(X_km_A, w_n_m)
  X_kn_B = get_X_kn(X_km_B, w_n_m)

  mean_mcd_over_all_k, res_penalty = compare_mel_spectrograms(
    X_kn_A, X_kn_B,
    s=s, D=D, aligning=aligning, align_target=align_target, remove_silence=remove_silence, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B
  )

  if aligned_here:
    assert res_penalty == 0
  else:
    assert penalty is None
    assert res_penalty is not None
    penalty = res_penalty

  return mean_mcd_over_all_k, penalty


def compare_mel_spectrograms(X_kn_A: np.ndarray, X_kn_B: np.ndarray, *, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["mel", "mfcc"] = "mel", remove_silence: Literal["no", "mel", "mfcc"] = "no", silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None) -> Tuple[float, float]:
  if len(X_kn_A) == 0:
    logger = getLogger(__name__)
    logger.warning("mel-spectrogram A is empty")
    return np.nan, np.nan

  if len(X_kn_B) == 0:
    logger = getLogger(__name__)
    logger.warning("mel-spectrogram B is empty")
    return np.nan, np.nan

  if not X_kn_A.shape[1] == X_kn_B.shape[1]:
    raise ValueError("both mel-spectrograms must have the same number of mel-bands")

  if remove_silence not in ["no", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'mel' or 'mfcc'")

  if align_target not in ["mel", "mfcc"]:
    raise ValueError("align_target must be 'mel' or 'mfcc'")

  if align_target == "mel" and remove_silence == "mfcc":
    raise ValueError(
        "cannot remove silence from MFCCs after both mel-spectrograms were aligned")

  N = X_kn_A.shape[1]
  if D > N:
    raise ValueError(f"D must be <= number of mel-bands ({N})")

  if remove_silence == "mel":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    X_kn_A = remove_silence_X_kn(X_kn_A, silence_threshold_A)
    X_kn_B = remove_silence_X_kn(X_kn_B, silence_threshold_B)

    if X_kn_A.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, mel-spectrogram A is empty")
      return np.nan, np.nan

    if X_kn_B.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, mel-spectrogram B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  penalty: float = None
  aligned_here: bool = False
  if align_target == "mel":
    X_kn_A, X_kn_B, penalty = align_X_kn(X_kn_A, X_kn_B, aligning)
    aligned_here = True
    align_target = "mfcc"
    aligning = "pad"

  # Shape: (N, #Frames)
  MC_X_ik = get_MC_X_ik(X_kn_A, N)
  MC_Y_ik = get_MC_X_ik(X_kn_B, N)

  remove_silence_mfcc = remove_silence == "mfcc"

  mean_mcd_over_all_k, res_penalty = compare_mfccs(
    MC_X_ik, MC_Y_ik,
    s=s, D=D, aligning=aligning, remove_silence=remove_silence_mfcc, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B
  )

  if aligned_here:
    assert res_penalty == 0
  else:
    assert penalty is None
    assert res_penalty is not None
    penalty = res_penalty

  return mean_mcd_over_all_k, penalty


def compare_mfccs(MC_X_ik: np.ndarray, MC_Y_ik: np.ndarray, *, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", remove_silence: bool = False, silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None) -> Tuple[float, float]:
  if MC_X_ik.shape[1] == 0:
    logger = getLogger(__name__)
    logger.warning("MFCCs A are empty")
    return np.nan, np.nan

  if MC_Y_ik.shape[1] == 0:
    logger = getLogger(__name__)
    logger.warning("MFCCs B are empty")
    return np.nan, np.nan

  if not MC_X_ik.shape[0] == MC_Y_ik.shape[0]:
    raise ValueError("both MFCCs must have the same number of coefficients")

  if not 0 <= s < D:
    raise ValueError("s must be in [0, D)")

  if D < 1:
    raise ValueError("D must be >= 1")

  if aligning not in ["pad", "dtw"]:
    raise ValueError("aligning must be 'pad' or 'dtw'")

  if remove_silence:
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    MC_X_ik = remove_silence_MC_X_ik(MC_X_ik, silence_threshold_A)
    MC_Y_ik = remove_silence_MC_X_ik(MC_Y_ik, silence_threshold_B)

    if MC_X_ik.shape[1] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, MFCCs A are empty")
      return np.nan, np.nan

    if MC_Y_ik.shape[1] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, MFCCs B are empty")
      return np.nan, np.nan

  MC_X_ik, MC_Y_ik, penalty = align_MC(MC_X_ik, MC_Y_ik, aligning)

  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)

  return mean_mcd_over_all_k, penalty
