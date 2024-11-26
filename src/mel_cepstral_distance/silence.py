
import numpy as np
import numpy.typing as npt

from mel_cepstral_distance.helper import amp_to_mag


def remove_silence_rms(audio_signal: np.ndarray, threshold_rms: float, min_silence_samples: int = 256):
  assert 0 <= threshold_rms <= 1
  if threshold_rms == 0:
    return audio_signal
  non_silent_audio = []

  start = 0
  while start < len(audio_signal):
    end = min(start + min_silence_samples, len(audio_signal))
    segment = audio_signal[start:end]

    rms_value = np.sqrt(np.mean(segment**2))

    if rms_value >= threshold_rms:
      non_silent_audio.append(segment)

    start = end

  if non_silent_audio:
    return np.concatenate(non_silent_audio)
  return np.array([], dtype=np.float32)


def get_loudness_vals_X_km(X_km: npt.NDArray[np.complex128]) -> np.ndarray:
  mel_energy = np.mean(amp_to_mag(X_km), axis=1)
  return mel_energy


def detect_non_silence_in_X_km(X_km: npt.NDArray[np.complex128], silence_threshold: float) -> np.ndarray:
  mel_energy = get_loudness_vals_X_km(X_km)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_X_km(X_km: npt.NDArray[np.complex128], silence_threshold: float) -> npt.NDArray[np.complex128]:
  frames = detect_non_silence_in_X_km(X_km, silence_threshold)
  X_km = X_km[frames, :]
  return X_km


def get_loudness_vals_X_kn(X_kn: np.ndarray) -> np.ndarray:
  mel_energy = np.mean(X_kn, axis=1)
  return mel_energy


def detect_non_silence_in_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  mel_energy = get_loudness_vals_X_kn(X_kn)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_X_kn(X_kn: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_X_kn(X_kn, silence_threshold)
  X_kn = X_kn[frames, :]
  return X_kn


def get_loudness_vals_MC_X_ik(MC_X_ik: np.ndarray) -> np.ndarray:
  mel_energy = MC_X_ik[0, :]
  return mel_energy


def detect_non_silence_in_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  # MC_X_ik - Shape: (# MFCC coefficients, time_frames)
  assert len(MC_X_ik) > 0, "Expected non-empty MFCC matrix"
  mel_energy = get_loudness_vals_MC_X_ik(MC_X_ik)
  non_silent_frame_indices = np.where(mel_energy >= silence_threshold)[0]
  return non_silent_frame_indices


def remove_silence_MC_X_ik(MC_X_ik: np.ndarray, silence_threshold: float) -> np.ndarray:
  frames = detect_non_silence_in_MC_X_ik(MC_X_ik, silence_threshold)
  MC_X_ik = MC_X_ik[:, frames]
  return MC_X_ik
