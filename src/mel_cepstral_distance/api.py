
import tempfile
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Literal, Tuple

import numpy as np
from scipy.io import wavfile

from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MC_X_ik_fast,
                                               get_MCD_k, get_MCD_k_fast, get_penalty, get_w_n_m,
                                               get_X_km, get_X_kn, get_X_kn_fast)
from mel_cepstral_distance.dtw import align_1d_sequences_using_dtw, align_2d_sequences_using_dtw
from mel_cepstral_distance.helper import (detect_non_silence_in_MC_X_ik,
                                          detect_non_silence_in_mfccs, fill_with_zeros_1d,
                                          fill_with_zeros_2d, ms_to_samples, norm_audio,
                                          plot_MC_X_ik, plot_X_km, plot_X_kn,
                                          remove_silence_from_spec, remove_silence_rms,
                                          resample_if_necessary, samples_to_ms)


def compare_audio_files(audioA: Path, audioB: Path, *, sample_rate: int = 8000, n_fft: float = 32, win_len: float = 32, hop_len: float = 16, window: Literal["hamming", "hanning"] = "hanning", low_freq: int = 0, high_freq: 4000, N: int = 20, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["sig", "spec", "mel", "mfcc"] = "spec", remove_silence: Literal["no", "sig", "spec", "mel"] = "spec", silence_threshold_A: float = 0.05, silence_threshold_B: float = 0.05, norm_sig: bool = True, debug: bool = False) -> None:
  """
  - silence is removed before alignment
  - high freq is max sr/2
  - n_fft should be equal to win_len
  - n_fft should be a power of 2 in samples
  - mel -8
  - spec 30 (sum) oder 0.05 (mean)
  - TODO check that align target is after sil rem target
  """
  logger = getLogger(__name__)
  start = perf_counter()
  penalty: float = 0

  log_dir = Path(tempfile.TemporaryDirectory(prefix="mel_cepstral_distance").name)
  log_dir.mkdir(parents=True, exist_ok=False)
  log_dir = Path("/tmp/mel_cepstral_distance")
  log_dir.mkdir(parents=True, exist_ok=True)

  logger.info("-- Reading audio files --")
  sr1, signalA = wavfile.read(audioA)
  logger.info(f"A -> Read '{audioA}' with sampling rate {sr1} in {perf_counter() - start:.2f}s")

  p = perf_counter()
  sr2, signalB = wavfile.read(audioB)
  logger.info(f"B -> Read '{audioB}' with sampling rate {sr2} in {perf_counter() - p:.2f}s")

  if norm_sig:
    logger.info("-- Normalizing audio signals --")
    signalA = norm_audio(signalA)
    signalB = norm_audio(signalB)
  else:
    logger.info("Not normalizing audio signals")

  logger.info("-- Resampling audio signals if necessary --")
  if sr1 != sample_rate:
    logger.info(f"A -> Resampling from {sr1}Hz to {sample_rate}Hz")
    signalA = resample_if_necessary(signalA, sr1, sample_rate)
  else:
    logger.info(f"A -> No resampling necessary because it is already at {sample_rate}Hz")

  if sr2 != sample_rate:
    logger.info(f"B -> Resampling from {sr2}Hz to {sample_rate}Hz")
    signalB = resample_if_necessary(signalB, sr2, sample_rate)
  else:
    logger.info(f"B -> No resampling necessary because it is already at {sample_rate}Hz")

  win_len_samples = ms_to_samples(win_len, sample_rate)

  if remove_silence == "sig" and silence_threshold_A > 0:
    logger.info("-- Removing silence from audio signals --")
    logger.info(f"Silence threshold: {silence_threshold_A}RMS")
    former_len_A = len(signalA)
    signalA = remove_silence_rms(
      signalA, silence_threshold_A,
      min_silence_samples=win_len_samples
    )
    if former_len_A != len(signalA):
      logger.info(
        f"A -> Removed {former_len_A - len(signalA)} silence samples ({samples_to_ms(former_len_A - len(signalA), sample_rate)}ms)")
    else:
      logger.info("A -> No silence samples removed")

    former_len_B = len(signalB)
    signalB = remove_silence_rms(
      signalB, silence_threshold_A,
      min_silence_samples=win_len_samples
    )
    if former_len_B != len(signalB):
      logger.info(
        f"B -> Removed {former_len_B - len(signalB)} silence samples ({samples_to_ms(former_len_B - len(signalB), sample_rate)}ms)")
    else:
      logger.info("B -> No silence samples removed")

  if align_target == "sig":
    logger.info("-- Aligning audio signals --")
    former_len_A = len(signalA)
    former_len_B = len(signalB)
    signalA, signalB, penalty = align_samples_1d(signalA, signalB, aligning)

  # Calculate STFTs (= spectrograms)
  logger.info("-- Calculating spectrograms using STFT --")

  hop_len_samples = ms_to_samples(hop_len, sample_rate)
  logger.info(f"Parameter hop_len: {hop_len_samples} samples ({hop_len}ms)")
  logger.info(f"Parameter win_len: {win_len_samples} samples ({win_len}ms)")

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  logger.info(f"Parameter n_fft: {n_fft_samples} samples ({n_fft}ms)")

  n_fft_is_two_power = n_fft_samples & (n_fft_samples - 1) == 0
  if not n_fft_is_two_power:
    logger.warning(
      f"Parameter n_fft ({n_fft}ms / {n_fft_samples} samples) should be a power of 2 in samples for faster computation.")
  else:
    logger.info(
      f"Parameter n_fft ({n_fft_samples} samples) is a power of 2 in samples which speeds up computation.")

  if n_fft != win_len:
    logger.warning(f"Parameter n_fft ({n_fft}ms) should be equal to win_len ({win_len}ms).")
    if n_fft < win_len:
      logger.warning(f"Truncating windows to n_fft ({n_fft}ms).")
    else:
      assert win_len > n_fft
      logger.warning(f"Padding windows to n_fft ({n_fft}ms).")
  else:
    logger.info("Parameter n_fft is equal to win_len, so no padding or truncation is necessary.")

  # STFT - Shape: (#Frames, Bins)
  X_km_A = get_X_km(signalA, n_fft_samples, win_len_samples, hop_len_samples, window)
  X_km_B = get_X_km(signalB, n_fft_samples, win_len_samples, hop_len_samples, window)
  logger.info(
    f"A -> Calculated {len(X_km_A)} frames with {X_km_A.shape[1]} frequency bins for spectrogram")
  logger.info(f"A -> Min: {X_km_A.min()}, Mean: {X_km_A.mean()}, Max: {X_km_A.max()}")
  logger.info(
    f"B -> Calculated {len(X_km_B)} frames with {X_km_B.shape[1]} frequency bins for spectrogram")
  logger.info(f"B -> Min: {X_km_B.min()}, Mean: {X_km_B.mean()}, Max: {X_km_B.max()}")
  if debug:
    fig = plot_X_km(X_km_A, sample_rate, hop_len_samples)
    fig.savefig(log_dir / "A_spectrogram.png")
    logger.info(f"Saved spectrogram of A to '{log_dir / 'A_spectrogram.png'}'")
    fig = plot_X_km(X_km_B, sample_rate, hop_len_samples)
    fig.savefig(log_dir / "B_spectrogram.png")
    logger.info(f"Saved spectrogram of B to '{log_dir / 'B_spectrogram.png'}'")

  if remove_silence == "spec":
    logger.info("-- Removing silence from Spectrograms --")
    logger.info(f"Silence threshold: {silence_threshold_A}")
    former_len_A = len(X_km_A)
    former_len_B = len(X_km_B)

    X_km_A = remove_silence_from_spec(X_km_A, silence_threshold_A)
    if former_len_A != len(X_km_A):
      logger.info(
        f"A -> Removed {former_len_A - len(X_km_A)}/{former_len_A} silence frames")
    else:
      logger.info("A -> No silence frames removed")

    X_km_B = remove_silence_from_spec(X_km_B, silence_threshold_A)
    if former_len_B != len(X_km_B):
      logger.info(
        f"B -> Removed {former_len_B - len(X_km_B)}/{former_len_B} silence frames")
    else:
      logger.info("B -> No silence frames removed")

  if align_target == "spec":
    logger.info("-- Aligning spectrograms --")
    X_km_A, X_km_B, penalty = align_frames_2d(X_km_A.T, X_km_B.T, aligning)
    X_km_A = X_km_A.T
    X_km_B = X_km_B.T

  logger.info("-- Calculating Mel-filterbank --")
  # Mel-Bank - Shape: (N, #Frames)
  w_n_m = get_w_n_m(sample_rate, n_fft_samples, N, low_freq, high_freq)

  logger.info("-- Calculating Mel-spectrograms --")
  # Mel-Spectrogram - Shape: (#Frames, #N)
  X_kn_A = get_X_kn_fast(X_km_A, w_n_m)
  X_kn_B = get_X_kn_fast(X_km_B, w_n_m)
  logger.info(
    f"A -> Calculated {len(X_kn_A)} frames with {X_kn_A.shape[1]} mel-bins for mel-spectrogram")
  logger.info(f"A -> Min: {X_kn_A.min()}, Mean: {X_kn_A.mean()}, Max: {X_kn_A.max()}")
  logger.info(
    f"B -> Calculated {len(X_kn_B)} frames with {X_kn_B.shape[1]} mel-bins for mel-spectrogram")
  logger.info(f"B -> Min: {X_kn_B.min()}, Mean: {X_kn_B.mean()}, Max: {X_kn_B.max()}")

  if debug:
    fig = plot_X_kn(X_kn_A, low_freq, high_freq)
    fig.savefig(log_dir / "A_mel_spectrogram.png")
    logger.info(f"Saved mel-spectrogram of A to '{log_dir / 'A_mel_spectrogram.png'}'")
    fig = plot_X_kn(X_kn_B, low_freq, high_freq)
    fig.savefig(log_dir / "B_mel_spectrogram.png")
    logger.info(f"Saved mel-spectrogram of B to '{log_dir / 'B_mel_spectrogram.png'}'")

  if remove_silence == "mel":
    logger.info("-- Removing silence from Mel-spectrograms --")
    logger.info(f"Silence threshold: {silence_threshold_A}dB")
    former_len_A = len(X_kn_A)
    former_len_B = len(X_kn_B)

    X_kn_A = remove_silence_from_spec(
      X_kn_A, silence_threshold_A,
    )
    if former_len_A != len(X_kn_A):
      logger.info(
        f"A -> Removed {former_len_A - len(X_kn_A)}/{former_len_A} silence frames")
    else:
      logger.info("A -> No silence frames removed")

    non_sil_frames_A = detect_non_silence_in_mfccs(X_kn_A, silence_threshold_A)
    logger.info(f"A -> Detected {len(X_kn_A) - non_sil_frames_A} silence frames")

    X_kn_B = remove_silence_from_spec(
      X_kn_B, silence_threshold_B,
    )
    if former_len_B != len(X_kn_B):
      logger.info(
        f"B -> Removed {former_len_B - len(X_kn_B)}/{former_len_B} silence frames")
    else:
      logger.info("B -> No silence frames removed")

  if align_target == "mel":
    logger.info("-- Aligning Mel-spectrograms --")
    X_kn_A, X_kn_B, penalty = align_frames_2d(X_kn_A.T, X_kn_B.T, aligning)
    X_kn_A = X_kn_A.T
    X_kn_B = X_kn_B.T

  # Calculate Mel Cepstral Coefficients
  logger.info("-- Calculating Mel Cepstral Coefficients --")
  # Shape: (N, #Frames)
  MC_X_ik = get_MC_X_ik_fast(X_kn_A, N)
  MC_Y_ik = get_MC_X_ik_fast(X_kn_B, N)
  logger.info(f"Calculated {N} MFCCs")
  logger.info(f"A -> Min: {MC_X_ik.min()}, Mean: {MC_X_ik.mean()}, Max: {MC_X_ik.max()}")
  logger.info(f"B -> Min: {MC_Y_ik.min()}, Mean: {MC_Y_ik.mean()}, Max: {MC_Y_ik.max()}")

  if debug:
    fig = plot_MC_X_ik(MC_X_ik)
    fig.savefig(log_dir / "A_mfcc.png")
    logger.info(f"Saved MFCC of A to '{log_dir / 'A_mfcc.png'}'")
    fig = plot_MC_X_ik(MC_Y_ik)
    fig.savefig(log_dir / "B_mfcc.png")
    logger.info(f"Saved MFCC of B to '{log_dir / 'B_mfcc.png'}'")

  if remove_silence == "mfcc":
    logger.info("-- Removing silence from MFCCs --")
    logger.info(f"Silence threshold: {silence_threshold_A}dB")
    former_len_A = MC_X_ik.shape[1]
    former_len_B = MC_Y_ik.shape[1]

    MC_X_ik_non_silent_idx = detect_non_silence_in_MC_X_ik(
      MC_X_ik, silence_threshold_A,
    )
    silent_frames_A = MC_X_ik.shape[1] - len(MC_X_ik_non_silent_idx)

    if silent_frames_A > 0:
      MC_X_ik = MC_X_ik[:, MC_X_ik_non_silent_idx]
      logger.info(
        f"A -> Removed {silent_frames_A} silence frames ({samples_to_ms(silent_frames_A * hop_len_samples, sample_rate)/1000:.0f}s)")

      if debug:
        fig = plot_X_km(X_km_A[MC_X_ik_non_silent_idx], sample_rate, hop_len_samples)
        fig.savefig(log_dir / "A_spectrogram_silence_removed.png")
        logger.info(
          f"Saved spectrogram of A with silence removed to '{log_dir / 'A_spectrogram_silence_removed.png'}'")

        fig = plot_X_kn(X_kn_A[MC_X_ik_non_silent_idx], low_freq, high_freq)
        fig.savefig(log_dir / "A_mel_spectrogram_silence_removed.png")
        logger.info(
          f"Saved mel-spectrogram of A with silence removed to '{log_dir / 'A_mel_spectrogram_silence_removed.png'}'")

        fig = plot_MC_X_ik(MC_X_ik)
        fig.savefig(log_dir / "A_mfcc_silence_removed.png")
        logger.info(
          f"Saved MFCC of A with silence removed to '{log_dir / 'A_mfcc_silence_removed.png'}'")

    else:
      logger.info("A -> No silence frames removed")

    MC_Y_ik_non_silent_idx = detect_non_silence_in_MC_X_ik(
      MC_Y_ik, silence_threshold_B,
    )
    silent_frames_B = MC_Y_ik.shape[1] - len(MC_Y_ik_non_silent_idx)

    if silent_frames_B > 0:
      MC_Y_ik = MC_Y_ik[:, MC_Y_ik_non_silent_idx]
      logger.info(
        f"B -> Removed {silent_frames_B} silence frames ({samples_to_ms(silent_frames_B * hop_len_samples, sample_rate)/1000:.0f}s)")

      if debug:
        fig = plot_X_km(X_km_B[MC_Y_ik_non_silent_idx], sample_rate, hop_len_samples)
        fig.savefig(log_dir / "B_spectrogram_silence_removed.png")
        logger.info(
          f"Saved spectrogram of B with silence removed to '{log_dir / 'B_spectrogram_silence_removed.png'}'")

        fig = plot_X_kn(X_kn_B[MC_Y_ik_non_silent_idx], low_freq, high_freq)
        fig.savefig(log_dir / "B_mel_spectrogram_silence_removed.png")
        logger.info(
          f"Saved mel-spectrogram of B with silence removed to '{log_dir / 'B_mel_spectrogram_silence_removed.png'}'")

        fig = plot_MC_X_ik(MC_Y_ik)
        fig.savefig(log_dir / "B_mfcc_silence_removed.png")
        logger.info(
          f"Saved MFCC of B with silence removed to '{log_dir / 'B_mfcc_silence_removed.png'}'")
    else:
      logger.info("B -> No silence frames removed")

  if align_target == "mfcc":
    logger.info("-- Aligning MFCCs --")
    MC_X_ik, MC_Y_ik, penalty = align_frames_2d(MC_X_ik, MC_Y_ik, aligning)

    if debug:
      fig = plot_MC_X_ik(MC_X_ik)
      fig.savefig(log_dir / "A_mfcc_aligned.png")
      logger.info(f"Saved aligned MFCC of A to '{log_dir / 'A_mfcc_aligned.png'}'")
      fig = plot_MC_X_ik(MC_Y_ik)
      fig.savefig(log_dir / "B_mfcc_aligned.png")
      logger.info(f"Saved aligned MFCC of B to '{log_dir / 'B_mfcc_aligned.png'}'")

  logger.info("-- Calculating Mel Cepstral Distance --")
  logger.info(f"Parameter s (first MFCC): {s}")
  logger.info(f"Parameter D (last MFCC): {D}")
  MCD_k = get_MCD_k_fast(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)
  logger.info(f"Mean MCD over all frames: {mean_mcd_over_all_k}")

  logger.info("-- Computation finished --")
  logger.info(f"Computation took {(perf_counter() - start)*1000:.0f}ms")
  if debug:
    logger.info(f"Debug information saved to '{log_dir}'")
  return mean_mcd_over_all_k, penalty


def align_samples_1d(seqA: np.ndarray, seqB: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  logger = getLogger(__name__)
  former_len_A = seqA.shape[1]
  former_len_B = seqB.shape[1]
  if aligning == "dtw":
    seqA, seqB = align_1d_sequences_using_dtw(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      logger.info(f"A -> Aligned from {former_len_A} to {seqA.shape[1]} samples")
    else:
      logger.info("No alignment necessary for A")

    if former_len_B != seqB.shape[1]:
      logger.info(f"B -> Aligned from {former_len_B} to {seqB.shape[1]} samples")
    else:
      logger.info("B -> No alignment necessary")
  else:
    assert aligning == "pad"
    seqA, seqB = fill_with_zeros_1d(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      logger.info(f"A -> Padded from {former_len_A} to {seqA.shape[1]} samples to match B")

    if former_len_B != seqB.shape[1]:
      logger.info(f"B -> Padded from {former_len_B} to {seqB.shape[1]} samples to match A")

  assert seqA.shape[1] == seqB.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, seqA.shape[1])
  return seqA, seqB, penalty


def align_frames_2d(seqA: np.ndarray, seqB: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert seqA.shape[0] == seqB.shape[0]
  logger = getLogger(__name__)
  former_len_A = seqA.shape[1]
  former_len_B = seqB.shape[1]
  if aligning == "dtw":
    seqA, seqB = align_2d_sequences_using_dtw(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      logger.info(f"A -> Aligned from {former_len_A} to {seqA.shape[1]} frames")
    else:
      logger.info("A -> No alignment necessary")

    if former_len_B != seqB.shape[1]:
      logger.info(f"B -> Aligned from {former_len_B} to {seqB.shape[1]} frames")
    else:
      logger.info("B -> No alignment necessary")
  else:
    assert aligning == "pad"
    seqA, seqB = fill_with_zeros_2d(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      logger.info(f"A -> Padded from {former_len_A} to {seqA.shape[1]} frames to match B")

    if former_len_B != seqB.shape[1]:
      logger.info(f"B -> Padded from {former_len_B} to {seqB.shape[1]} frames to match A")

  assert seqA.shape[1] == seqB.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, seqA.shape[1])
  logger.info(f"Alignment penalty: {penalty:.4f}")
  return seqA, seqB, penalty
