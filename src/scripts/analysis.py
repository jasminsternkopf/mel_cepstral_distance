
import tempfile
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from mel_cepstral_distance.alignment import align_2d_sequences_using_dtw
from mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_w_n_m,
                                               get_X_km, get_X_kn)
from mel_cepstral_distance.helper import (detect_non_silence_in_MC_X_ik, detect_non_silence_in_X_km,
                                          detect_non_silence_in_X_kn,
                                          extract_extract_frames_from_signal, fill_with_zeros_2d,
                                          get_penalty, ms_to_samples, norm_audio_signal,
                                          plot_MC_X_ik, plot_X_km, plot_X_kn, remove_silence_rms,
                                          resample_if_necessary, samples_to_ms,
                                          stack_images_vertically)


def compare_audio_files_extended(audio_A: Path, audio_B: Path, *, sample_rate: int = 8000, n_fft: float = 32, win_len: float = 32, hop_len: float = 16, window: Literal["hamming", "hanning"] = "hanning", low_freq: int = 0, high_freq: 4000, N: int = 20, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["spec", "mel", "mfcc"] = "spec", remove_silence: Literal["no", "sig", "spec", "mel", "mfcc"] = "spec", silence_threshold_A: float = 0.05, silence_threshold_B: float = 0.05, norm_sig: bool = True, custom_save_dir: Optional[Path] = None) -> None:
  logger = getLogger(__name__)
  start = perf_counter()

  step = 0

  if remove_silence == "mel":
    if align_target == "spec":
      raise ValueError(
        "Cannot remove silence from mel-spectrogram after both spectrograms were aligned")
  if remove_silence == "mfcc":
    if align_target == "spec":
      raise ValueError(
        "Cannot remove silence from MFCCs after both spectrograms were aligned")
    if align_target == "mel":
      raise ValueError(
        "Cannot remove silence from MFCCs after both mel-spectrograms were aligned")

  penalty: float = 0

  if custom_save_dir is None:
    log_dir = Path(tempfile.TemporaryDirectory(prefix="mel_cepstral_distance").name)
    log_dir.mkdir(parents=True, exist_ok=False)
  else:
    log_dir = custom_save_dir
    log_dir.mkdir(parents=True, exist_ok=True)

  step += 1
  print(f"-- ({step}) Reading audio files --")
  sr1, signalA = wavfile.read(audio_A)
  print(f"A -> Read '{audio_A}' with sampling rate {sr1} in {perf_counter() - start:.2f}s")

  p = perf_counter()
  sr2, signalB = wavfile.read(audio_B)
  print(f"B -> Read '{audio_B}' with sampling rate {sr2} in {perf_counter() - p:.2f}s")

  if norm_sig:
    step += 1
    print(f"-- ({step}) Normalizing audio signals --")
    signalA = norm_audio_signal(signalA)
    signalB = norm_audio_signal(signalB)
    wavfile.write(log_dir / f"{step}_A_normalized.wav", sr1, signalA)
    wavfile.write(log_dir / f"{step}_B_normalized.wav", sr2, signalB)
  else:
    print("Not normalizing audio signals")

  step += 1
  print(f"-- ({step}) Resampling audio signals if necessary --")
  if sr1 != sample_rate:
    print(f"A -> Resampling from {sr1}Hz to {sample_rate}Hz")
    signalA = resample_if_necessary(signalA, sr1, sample_rate)
    wavfile.write(log_dir / f"{step}_A_resampled.wav", sample_rate, signalA)
  else:
    print(f"A -> No resampling necessary because it is already at {sample_rate}Hz")

  if sr2 != sample_rate:
    print(f"B -> Resampling from {sr2}Hz to {sample_rate}Hz")
    signalB = resample_if_necessary(signalB, sr2, sample_rate)
    wavfile.write(log_dir / f"{step}_B_resampled.wav", sample_rate, signalB)
  else:
    print(f"B -> No resampling necessary because it is already at {sample_rate}Hz")

  win_len_samples = ms_to_samples(win_len, sample_rate)

  if remove_silence == "sig":
    step += 1
    print(f"-- ({step}) Removing silence from audio signals --")
    print(f"Silence threshold A: {silence_threshold_A}RMS")
    print(f"Silence threshold B: {silence_threshold_B}RMS")

    former_len_A = len(signalA)
    signalA = remove_silence_rms(
      signalA, silence_threshold_A,
      min_silence_samples=win_len_samples
    )
    if former_len_A != len(signalA):
      print(
        f"A -> Removed {former_len_A - len(signalA)} silence samples ({samples_to_ms(former_len_A - len(signalA), sample_rate)}ms)")
      wavfile.write(log_dir / f"{step}_A_silence_removed.wav", sample_rate, signalA)
    else:
      print("A -> No silence samples removed")

    former_len_B = len(signalB)
    signalB = remove_silence_rms(
      signalB, silence_threshold_B,
      min_silence_samples=win_len_samples
    )
    if former_len_B != len(signalB):
      print(
        f"B -> Removed {former_len_B - len(signalB)} silence samples ({samples_to_ms(former_len_B - len(signalB), sample_rate)}ms)")
      wavfile.write(log_dir / f"{step}_B_silence_removed.wav", sample_rate, signalB)
    else:
      print("B -> No silence samples removed")

  stack_pipeline_A = []
  stack_pipeline_B = []

  # Calculate STFTs (= spectrograms)
  step += 1
  print(f"-- ({step}) Calculating spectrograms using STFT --")

  hop_len_samples = ms_to_samples(hop_len, sample_rate)
  print(f"Parameter hop_len: {hop_len_samples} samples ({hop_len}ms)")
  print(f"Parameter win_len: {win_len_samples} samples ({win_len}ms)")

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  print(f"Parameter n_fft: {n_fft_samples} samples ({n_fft}ms)")

  n_fft_is_two_power = n_fft_samples & (n_fft_samples - 1) == 0
  if not n_fft_is_two_power:
    logger.warning(
      f"Parameter n_fft ({n_fft}ms / {n_fft_samples} samples) should be a power of 2 in samples for faster computation.")
  else:
    print(
      f"Parameter n_fft ({n_fft_samples} samples) is a power of 2 in samples which speeds up computation.")

  if n_fft != win_len:
    logger.warning(f"Parameter n_fft ({n_fft}ms) should be equal to win_len ({win_len}ms).")
    if n_fft < win_len:
      logger.warning(f"Truncating windows to n_fft ({n_fft}ms).")
    else:
      assert win_len > n_fft
      logger.warning(f"Padding windows to n_fft ({n_fft}ms).")
  else:
    print("Parameter n_fft is equal to win_len, so no padding or truncation is necessary.")

  # STFT - Shape: (#Frames, Bins)
  X_km_A = get_X_km(signalA, n_fft_samples, win_len_samples, hop_len_samples, window)
  X_km_B = get_X_km(signalB, n_fft_samples, win_len_samples, hop_len_samples, window)

  print(
    f"A -> Calculated {len(X_km_A)} frames with {X_km_A.shape[1]} frequency bins for spectrogram")
  print(f"A -> Min: {X_km_A.min()}, Mean: {X_km_A.mean()}, Max: {X_km_A.max()}")
  print(
    f"B -> Calculated {len(X_km_B)} frames with {X_km_B.shape[1]} frequency bins for spectrogram")
  print(f"B -> Min: {X_km_B.min()}, Mean: {X_km_B.mean()}, Max: {X_km_B.max()}")

  path_save_A_spectrogram = log_dir / f"{step}_A_spectrogram.png"
  np.save(log_dir / f"{step}_A_spectrogram.npy", X_km_A)
  fig = plot_X_km(X_km_A, sample_rate, "A (input)")
  fig.savefig(path_save_A_spectrogram)
  plt.close()
  stack_pipeline_A.append(path_save_A_spectrogram)

  path_save_B_spectrogram = log_dir / f"{step}_B_spectrogram.png"
  np.save(log_dir / f"{step}_B_spectrogram.npy", X_km_B)
  fig = plot_X_km(X_km_B, sample_rate, "B (input)")
  fig.savefig(path_save_B_spectrogram)
  plt.close()
  stack_pipeline_B.append(path_save_B_spectrogram)

  if remove_silence == "spec":
    step += 1
    print(f"-- ({step}) Removing silence from spectrograms --")

    print(f"Silence threshold A: {silence_threshold_A}")
    X_km_A_non_silent_idx = detect_non_silence_in_X_km(X_km_A, silence_threshold_A)
    len_silent_frames_A = len(X_km_A) - len(X_km_A_non_silent_idx)

    if len_silent_frames_A > 0:
      former_len_A = len(X_km_A)
      X_km_A = X_km_A[X_km_A_non_silent_idx, :]
      print(
        f"A -> Removed {len_silent_frames_A}/{former_len_A} silence frames")

      np.save(log_dir / f"{step}_A_spectrogram_silence_removed.npy", X_km_A)
      fig = plot_X_km(X_km_A, sample_rate, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_A_spectrogram_silence_removed.png")
      plt.close()
      stack_pipeline_A.append(log_dir / f"{step}_A_spectrogram_silence_removed.png")

      stack_images_vertically([
        path_save_A_spectrogram,
        log_dir / f"{step}_A_spectrogram_silence_removed.png",
      ], log_dir / f"{step}_A_silence_removed.png")

    else:
      print("A -> No silence frames removed")

    print(f"Silence threshold B: {silence_threshold_B}")
    X_km_B_non_silent_idx = detect_non_silence_in_X_km(X_km_B, silence_threshold_B)
    len_silent_frames_B = len(X_km_B) - len(X_km_B_non_silent_idx)

    if len_silent_frames_B > 0:
      former_len_B = len(X_km_B)
      X_km_B = X_km_B[X_km_B_non_silent_idx, :]
      print(
        f"B -> Removed {len_silent_frames_B}/{former_len_B} silence frames")

      np.save(log_dir / f"{step}_B_spectrogram_silence_removed.npy", X_km_B)
      fig = plot_X_km(X_km_B, sample_rate, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_B_spectrogram_silence_removed.png")
      plt.close()
      stack_pipeline_B.append(log_dir / f"{step}_B_spectrogram_silence_removed.png")

      stack_images_vertically([
        path_save_B_spectrogram,
        log_dir / f"{step}_B_spectrogram_silence_removed.png",
      ], log_dir / f"{step}_B_silence_removed.png")
    else:
      print("B -> No silence frames removed")

  if align_target == "spec":
    step += 1
    print(f"-- ({step}) Aligning spectrograms --")
    X_km_A, X_km_B, penalty = align_frames_2d(X_km_A.T, X_km_B.T, aligning)
    X_km_A = X_km_A.T
    X_km_B = X_km_B.T

    np.save(log_dir / f"{step}_A_spectrogram_aligned.npy", X_km_A)
    fig = plot_X_km(X_km_A, sample_rate, "A (aligned)")
    fig.savefig(log_dir / f"{step}_A_spectrogram_aligned.png")
    plt.close()
    stack_pipeline_A.append(log_dir / f"{step}_A_spectrogram_aligned.png")

    np.save(log_dir / f"{step}_B_spectrogram_aligned.npy", X_km_B)
    fig = plot_X_km(X_km_B, sample_rate, "B (aligned)")
    fig.savefig(log_dir / f"{step}_B_spectrogram_aligned.png")
    plt.close()
    stack_pipeline_B.append(log_dir / f"{step}_B_spectrogram_aligned.png")

  step += 1
  print(f"-- ({step}) Calculating Mel-filterbank --")
  # Mel-Bank - Shape: (N, #Frames)
  w_n_m = get_w_n_m(sample_rate, n_fft_samples, N, low_freq, high_freq)

  step += 1
  print(f"-- ({step}) Calculating Mel-spectrograms --")
  # Mel-Spectrogram - Shape: (#Frames, #N)
  X_kn_A = get_X_kn(X_km_A, w_n_m)
  X_kn_B = get_X_kn(X_km_B, w_n_m)

  print(
    f"A -> Calculated {len(X_kn_A)} frames with {X_kn_A.shape[1]} mel-bins for mel-spectrogram")
  print(f"A -> Min: {X_kn_A.min()}, Mean: {X_kn_A.mean()}, Max: {X_kn_A.max()}")
  print(
    f"B -> Calculated {len(X_kn_B)} frames with {X_kn_B.shape[1]} mel-bins for mel-spectrogram")
  print(f"B -> Min: {X_kn_B.min()}, Mean: {X_kn_B.mean()}, Max: {X_kn_B.max()}")

  np.save(log_dir / f"{step}_A_mel_spectrogram.npy", X_kn_A)
  fig = plot_X_kn(X_kn_A, low_freq, high_freq, "A (input)")
  path_mel_spectrogram_A = log_dir / f"{step}_A_mel_spectrogram.png"
  fig.savefig(path_mel_spectrogram_A)
  plt.close()
  stack_pipeline_A.append(path_mel_spectrogram_A)

  np.save(log_dir / f"{step}_B_mel_spectrogram.npy", X_kn_B)
  fig = plot_X_kn(X_kn_B, low_freq, high_freq, "B (input)")
  path_mel_spectrogram_B = log_dir / f"{step}_B_mel_spectrogram.png"
  fig.savefig(path_mel_spectrogram_B)
  plt.close()
  stack_pipeline_B.append(path_mel_spectrogram_B)

  if remove_silence == "mel":
    step += 1
    print(f"-- ({step}) Removing silence from Mel-spectrograms --")

    print(f"Silence threshold A: {silence_threshold_A}dB")
    X_kn_A_non_silent_idx = detect_non_silence_in_X_kn(X_kn_A, silence_threshold_A)
    len_silent_frames_A = len(X_kn_A) - len(X_kn_A_non_silent_idx)

    if len_silent_frames_A > 0:
      former_len_A = len(X_kn_A)
      X_kn_A = X_kn_A[X_kn_A_non_silent_idx, :]
      print(
        f"A -> Removed {len_silent_frames_A}/{former_len_A} silence frames")

      signalA_silence_removed = extract_extract_frames_from_signal(
        signalA, X_kn_A_non_silent_idx, hop_len_samples)
      wavfile.write(log_dir / f"{step}_1_A_silence_removed.wav",
                    sample_rate, signalA_silence_removed)

      fig = plot_X_km(X_km_A[X_kn_A_non_silent_idx, :], sample_rate, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_2_A_spectrogram_silence_removed.png")
      plt.close()

      np.save(log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.npy", X_kn_A)
      fig = plot_X_kn(X_kn_A, low_freq, high_freq, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.png")
      plt.close()
      stack_pipeline_A.append(log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.png")

      stack_images_vertically([
        path_mel_spectrogram_A,
        log_dir / f"{step}_2_A_spectrogram_silence_removed.png",
        log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.png",
      ], log_dir / f"{step}_4_A_silence_removed.png")
    else:
      print("A -> No silence frames removed")

    print(f"Silence threshold B: {silence_threshold_B}dB")
    X_kn_B_non_silent_idx = detect_non_silence_in_X_kn(X_kn_B, silence_threshold_B)
    len_silent_frames_B = len(X_kn_B) - len(X_kn_B_non_silent_idx)

    if len_silent_frames_B > 0:
      former_len_B = len(X_kn_B)
      X_kn_B = X_kn_B[X_kn_B_non_silent_idx, :]
      print(
        f"B -> Removed {len_silent_frames_B}/{former_len_B} silence frames")

      signalB_silence_removed = extract_extract_frames_from_signal(
        signalB, X_kn_B_non_silent_idx, hop_len_samples)
      wavfile.write(log_dir / f"{step}_1_B_silence_removed.wav",
                    sample_rate, signalB_silence_removed)

      fig = plot_X_km(X_km_B[X_kn_B_non_silent_idx, :], sample_rate, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_2_B_spectrogram_silence_removed.png")
      plt.close()

      np.save(log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.npy", X_kn_B)
      fig = plot_X_kn(X_kn_B, low_freq, high_freq, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.png")
      plt.close()
      stack_pipeline_B.append(log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.png")

      stack_images_vertically([
        path_mel_spectrogram_B,
        log_dir / f"{step}_2_B_spectrogram_silence_removed.png",
        log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.png",
      ], log_dir / f"{step}_4_B_silence_removed.png")
    else:
      print("B -> No silence frames removed")

  if align_target == "mel":
    step += 1
    print(f"-- ({step}) Aligning Mel-spectrograms --")
    X_kn_A, X_kn_B, penalty = align_frames_2d(X_kn_A.T, X_kn_B.T, aligning)
    X_kn_A = X_kn_A.T
    X_kn_B = X_kn_B.T

    np.save(log_dir / f"{step}_A_mel_spectrogram_aligned.npy", X_kn_A)
    np.save(log_dir / f"{step}_B_mel_spectrogram_aligned.npy", X_kn_B)

    fig = plot_X_kn(X_kn_A, low_freq, high_freq, "A (aligned)")
    fig.savefig(log_dir / f"{step}_A_mel_spectrogram_aligned.png")
    plt.close()
    stack_pipeline_A.append(log_dir / f"{step}_A_mel_spectrogram_aligned.png")

    fig = plot_X_kn(X_kn_B, low_freq, high_freq, "B (aligned)")
    fig.savefig(log_dir / f"{step}_B_mel_spectrogram_aligned.png")
    plt.close()
    stack_pipeline_B.append(log_dir / f"{step}_B_mel_spectrogram_aligned.png")

    stack_images_vertically([
      log_dir / f"{step}_A_mel_spectrogram_aligned.png",
      log_dir / f"{step}_B_mel_spectrogram_aligned.png",
    ], log_dir / f"{step}_AB_mel_spectrogram_aligned.png")

  # Calculate Mel Cepstral Coefficients
  step += 1
  print(f"-- ({step}) Calculating Mel Cepstral Coefficients --")
  # Shape: (N, #Frames)
  MC_X_ik = get_MC_X_ik(X_kn_A, N)
  MC_Y_ik = get_MC_X_ik(X_kn_B, N)

  print(f"Calculated {N} MFCCs")
  print(f"A -> Min: {MC_X_ik.min()}, Mean: {MC_X_ik.mean()}, Max: {MC_X_ik.max()}")
  print(f"B -> Min: {MC_Y_ik.min()}, Mean: {MC_Y_ik.mean()}, Max: {MC_Y_ik.max()}")

  np.save(log_dir / f"{step}_A_mfcc.npy", MC_X_ik)
  np.save(log_dir / f"{step}_B_mfcc.npy", MC_Y_ik)

  fig = plot_MC_X_ik(MC_X_ik, "A (input)")
  fig.savefig(log_dir / f"{step}_A_mfcc.png")
  plt.close()
  stack_pipeline_A.append(log_dir / f"{step}_A_mfcc.png")

  fig = plot_MC_X_ik(MC_Y_ik, "B (input)")
  fig.savefig(log_dir / f"{step}_B_mfcc.png")
  plt.close()
  stack_pipeline_B.append(log_dir / f"{step}_B_mfcc.png")

  if remove_silence == "mfcc":
    step += 1
    print(f"-- ({step}) Removing silence from MFCCs --")
    print(f"Silence threshold: {silence_threshold_A}dB")

    non_silent_frames_A = detect_non_silence_in_MC_X_ik(MC_X_ik, silence_threshold_A)
    len_silent_frames_A = MC_X_ik.shape[1] - len(non_silent_frames_A)

    if len_silent_frames_A > 0:
      MC_X_ik = MC_X_ik[:, non_silent_frames_A]

      print(
        f"A -> Removed {len_silent_frames_A} silence frames ({samples_to_ms(len_silent_frames_A * hop_len_samples, sample_rate)/1000:.0f}s)")

      signalA_silence_removed = extract_extract_frames_from_signal(
        signalA, non_silent_frames_A, hop_len_samples)
      wavfile.write(log_dir / f"{step}_1_A_silence_removed.wav",
                    sample_rate, signalA_silence_removed)

      fig = plot_X_km(X_km_A[non_silent_frames_A], sample_rate, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_2_A_spectrogram_silence_removed.png")
      plt.close()

      fig = plot_X_kn(X_kn_A[non_silent_frames_A], low_freq, high_freq, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.png")
      plt.close()

      np.save(log_dir / f"{step}_4_A_mfcc_silence_removed.npy", MC_X_ik)
      fig = plot_MC_X_ik(MC_X_ik, "A (silence removed)")
      fig.savefig(log_dir / f"{step}_4_A_mfcc_silence_removed.png")
      plt.close()
      stack_pipeline_A.append(log_dir / f"{step}_4_A_mfcc_silence_removed.png")

      stack_images_vertically([
        log_dir / f"{step}_2_A_spectrogram_silence_removed.png",
        log_dir / f"{step}_3_A_mel_spectrogram_silence_removed.png",
        log_dir / f"{step}_4_A_mfcc_silence_removed.png",
      ], log_dir / f"{step}_5_A_silence_removed.png")
    else:
      print("A -> No silence frames removed")

    non_silent_frames_B = detect_non_silence_in_MC_X_ik(MC_Y_ik, silence_threshold_B)
    len_silent_frames_B = MC_Y_ik.shape[1] - len(non_silent_frames_B)

    if len_silent_frames_B > 0:
      MC_Y_ik = MC_Y_ik[:, non_silent_frames_B]

      print(
        f"B -> Removed {len_silent_frames_B} silence frames ({samples_to_ms(len_silent_frames_B * hop_len_samples, sample_rate)/1000:.0f}s)")

      signalB_silence_removed = extract_extract_frames_from_signal(
        signalB, non_silent_frames_B, hop_len_samples)
      wavfile.write(log_dir / f"{step}_1_B_silence_removed.wav",
                    sample_rate, signalB_silence_removed)

      fig = plot_X_km(X_km_B[non_silent_frames_B], sample_rate, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_2_B_spectrogram_silence_removed.png")
      plt.close()

      fig = plot_X_kn(X_kn_B[non_silent_frames_B], low_freq, high_freq, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.png")
      plt.close()

      np.save(log_dir / f"{step}_B_mfcc_silence_removed.npy", MC_Y_ik)
      fig = plot_MC_X_ik(MC_Y_ik, "B (silence removed)")
      fig.savefig(log_dir / f"{step}_4_B_mfcc_silence_removed.png")
      plt.close()
      stack_pipeline_B.append(log_dir / f"{step}_4_B_mfcc_silence_removed.png")

      stack_images_vertically([
        log_dir / f"{step}_2_B_spectrogram_silence_removed.png",
        log_dir / f"{step}_3_B_mel_spectrogram_silence_removed.png",
        log_dir / f"{step}_4_B_mfcc_silence_removed.png",
      ], log_dir / f"{step}_5_B_silence_removed.png")

    else:
      print("B -> No silence frames removed")

  if align_target == "mfcc":
    step += 1
    print(f"-- ({step}) Aligning MFCCs --")
    MC_X_ik, MC_Y_ik, penalty = align_frames_2d(MC_X_ik, MC_Y_ik, aligning)

    np.save(log_dir / f"{step}_A_mfcc_aligned.npy", MC_X_ik)
    np.save(log_dir / f"{step}_B_mfcc_aligned.npy", MC_Y_ik)

    fig = plot_MC_X_ik(MC_X_ik, "A (aligned)")
    fig.savefig(log_dir / f"{step}_A_mfcc_aligned.png")
    plt.close()
    stack_pipeline_A.append(log_dir / f"{step}_A_mfcc_aligned.png")

    fig = plot_MC_X_ik(MC_Y_ik, "B (aligned)")
    fig.savefig(log_dir / f"{step}_B_mfcc_aligned.png")
    plt.close()
    stack_pipeline_B.append(log_dir / f"{step}_B_mfcc_aligned.png")

  step += 1
  print(f"-- ({step}) Calculating Mel Cepstral Distance --")
  print(f"Parameter s (first MFCC): {s}")
  print(f"Parameter D (last MFCC): {D}")

  # Calculate Mel Cepstral Distance
  MCD_k = get_MCD_k(MC_X_ik, MC_Y_ik, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)

  print(f"Mean MCD over all frames: {mean_mcd_over_all_k}")

  step += 1
  print(f"-- ({step}) Computation finished --")
  print(f"Computation took {(perf_counter() - start)*1000:.0f}ms")

  stack_images_vertically(stack_pipeline_A, log_dir / f"{step}_A_pipeline.png")
  stack_images_vertically(stack_pipeline_B, log_dir / f"{step}_B_pipeline.png")
  print(f"Debug information saved to '{log_dir}'")

  return mean_mcd_over_all_k, penalty


def align_frames_2d(seqA: np.ndarray, seqB: np.ndarray, aligning: Literal["dtw", "pad"]) -> Tuple[np.ndarray, np.ndarray, float]:
  assert seqA.shape[0] == seqB.shape[0]
  former_len_A = seqA.shape[1]
  former_len_B = seqB.shape[1]
  if aligning == "dtw":
    seqA, seqB = align_2d_sequences_using_dtw(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      print(f"A -> Aligned from {former_len_A} to {seqA.shape[1]} frames")
    else:
      print("A -> No alignment necessary")

    if former_len_B != seqB.shape[1]:
      print(f"B -> Aligned from {former_len_B} to {seqB.shape[1]} frames")
    else:
      print("B -> No alignment necessary")
  else:
    assert aligning == "pad"
    seqA, seqB = fill_with_zeros_2d(seqA, seqB)

    if former_len_A != seqA.shape[1]:
      print(f"A -> Padded from {former_len_A} to {seqA.shape[1]} frames to match B")

    if former_len_B != seqB.shape[1]:
      print(f"B -> Padded from {former_len_B} to {seqB.shape[1]} frames to match A")

  assert seqA.shape[1] == seqB.shape[1]
  penalty = get_penalty(former_len_A, former_len_B, seqA.shape[1])
  print(f"Alignment penalty: {penalty:.4f}")
  return seqA, seqB, penalty
