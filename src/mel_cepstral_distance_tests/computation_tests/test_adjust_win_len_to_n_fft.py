import numpy as np

from mel_cepstral_distance.computation import adjust_win_len_to_n_fft


def test_padding_short_window():
  """Pads shorter frames to match n_fft."""
  windowed_frames = np.random.rand(3, 10)  # 3 Frames, 10 Samples
  n_fft = 16
  padded_frames, win_len = adjust_win_len_to_n_fft(windowed_frames, n_fft)

  assert padded_frames.shape == (3, n_fft), "Frames should be padded to match n_fft."
  assert np.all(padded_frames[:, 10:] == 0), "Padding should use zeros."
  assert win_len == n_fft, "win_len should equal n_fft after padding."


def test_truncating_long_window():
  """Truncates longer frames to match n_fft."""
  windowed_frames = np.random.rand(3, 20)  # 3 Frames, 20 Samples
  n_fft = 16
  truncated_frames, win_len = adjust_win_len_to_n_fft(windowed_frames, n_fft)

  assert truncated_frames.shape == (3, n_fft), "Frames should be truncated to match n_fft."
  assert np.all(truncated_frames == windowed_frames[:, :n_fft]
                ), "Truncated frames should match original up to n_fft."
  assert win_len == n_fft, "win_len should equal n_fft after truncation."


def test_no_change_when_equal():
  """Ensures frames remain unchanged when win_len == n_fft."""
  windowed_frames = np.random.rand(3, 16)  # 3 Frames, 16 Samples
  n_fft = 16
  unchanged_frames, win_len = adjust_win_len_to_n_fft(windowed_frames, n_fft)

  assert np.array_equal(
    unchanged_frames, windowed_frames), "Frames should remain unchanged when win_len == n_fft."
  assert unchanged_frames.shape == (3, n_fft), "Shape should remain the same."
  assert win_len == n_fft, "win_len should remain unchanged."


def test_edge_case_empty_frames():
  """Handles empty input frames correctly."""
  windowed_frames = np.empty((0, 10))  # No frames, 10 Samples
  n_fft = 16
  padded_frames, win_len = adjust_win_len_to_n_fft(windowed_frames, n_fft)

  assert padded_frames.shape == (0, n_fft), "Empty input should still return the correct shape."
  assert win_len == n_fft, "win_len should be set to n_fft."


def test_edge_case_single_frame():
  """Handles a single frame correctly."""
  windowed_frames = np.random.rand(1, 10)  # 1 Frame, 10 Samples
  n_fft = 16
  padded_frames, win_len = adjust_win_len_to_n_fft(windowed_frames, n_fft)

  assert padded_frames.shape == (1, n_fft), "Single frame should be padded correctly."
  assert win_len == n_fft, "win_len should equal n_fft after adjustment."
