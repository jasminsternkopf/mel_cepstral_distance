import numpy as np

from mel_cepstral_distance_analysis.helper import extract_frames_from_signal


def test_simple_signal():
  signal = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0])
  non_silent_frames = np.array([1, 3])
  hop_len_samples = 2
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([1.0, 2.0, 3.0, 4.0])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"


def test_single_non_silent_frame():
  signal = np.array([
    1.0, 1.1,
    0.0, 0.0,
    0.0
  ])
  non_silent_frames = np.array([0])
  hop_len_samples = 2
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([1.0, 1.1])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"


def test_all_non_silent_frames():
  signal = np.array([
    0.1, 0.2,
    0.3, 0.4,
    0.5, 0.6
  ])
  non_silent_frames = np.array([0, 1, 2])
  hop_len_samples = 2
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"


def test_empty_signal():
  signal = np.array([])
  non_silent_frames = np.array([0, 1])
  hop_len_samples = 2
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"


def test_single_sample_signal():
  signal = np.array([0.9])
  non_silent_frames = np.array([0])
  hop_len_samples = 1
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([0.9])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"


def test_overlapping_frames():
  signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
  non_silent_frames = np.array([0, 2])
  hop_len_samples = 3
  result = extract_frames_from_signal(signal, non_silent_frames, hop_len_samples)
  expected = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
  assert np.array_equal(result, expected), f"Test failed: {result} != {expected}"
