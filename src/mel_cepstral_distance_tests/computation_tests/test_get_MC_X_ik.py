import numpy as np

from mel_cepstral_distance.computation import get_MC_X_ik


def test_large_valid_input():
  X_kn = np.random.rand(10, 5)  # Random values for larger input
  M = 5
  result = get_MC_X_ik(X_kn, M)
  assert result.shape == (
    M, X_kn.shape[0]), f"Expected output shape {(M, X_kn.shape[0])}, but got {result.shape}."


def test_zero():
  X_kn = np.zeros((10, 5))
  M = 5
  result = get_MC_X_ik(X_kn, M)
  assert np.all(result == 0), f"Expected all zeros, but got {result}."


def test_basic_input_three_dim():
  X_kn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  M = 3
  expected = np.array([
    [-1.73205081e+00, -1.73205081e+00, -1.73205081e+00],
    [4.44089210e-16, 1.33226763e-15, 1.77635684e-15],
    [3.27685866e-15, 6.49248498e-15, 9.70811130e-15]
  ])
  result = get_MC_X_ik(X_kn, M)
  assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


def test_basic_input_two_dim():
  X_kn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  M = 2
  expected = np.array([
    [-7.07106781e-01, -7.07106781e-01, -7.07106781e-01],
    [-3.06161700e-16, -6.73555740e-16, -1.04094978e-15]
  ])
  result = get_MC_X_ik(X_kn, M)
  assert np.allclose(result, expected), f"Expected {expected}, but got {result}."


def test_basic_input_one_dim():
  X_kn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  M = 1
  expected = np.array([
    [6.1232340e-17, 2.4492936e-16, 4.2862638e-16]
  ])
  result = get_MC_X_ik(X_kn, M)
  assert np.allclose(result, expected), f"Expected {expected}, but got {result}."
