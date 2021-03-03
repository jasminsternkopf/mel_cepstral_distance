import librosa
import numpy as np

NO_OF_CRITICAL_BAND_FILTERS = 20
SAMPLING_RATE = 22050
FRAME_SIZE = 1024
HOP_SIZE = 256
NO_OF_COEFFICIENTS_PER_FRAME = 16


def cos_func(i, n):
  return np.cos((i + 1) * (n + 1 / 2) * np.pi / 20)


COS_MATRIX = np.fromfunction(cos_func, (NO_OF_COEFFICIENTS_PER_FRAME,
                                        NO_OF_CRITICAL_BAND_FILTERS), dtype=np.float64)
MEL_FILTER_BANK = librosa.filters.mel(sr=SAMPLING_RATE,
                                      n_fft=FRAME_SIZE,
                                      n_mels=NO_OF_CRITICAL_BAND_FILTERS,
                                      norm=None,
                                      dtype=np.float64,
                                      htk=True)
