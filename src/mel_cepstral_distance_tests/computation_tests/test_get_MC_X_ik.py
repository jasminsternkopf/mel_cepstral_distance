import numpy as np

from mel_cepstral_distance.computation import get_MC_X_ik


def get_MC_X_ik_from_paper(X_kn: np.ndarray, M: int) -> np.ndarray:
  """"
  Calculates the mel cepstrum coefficients of the mel spectrogram
  returns mel cepstrum with shape (M, #frames)
  """
  # K: total frame count
  # M: number of cepstral coefficients
  assert X_kn.ndim == 2, f"Expected a 2D array, but got {X_kn.ndim} dimensions"
  assert isinstance(M, int) and M > 0, "M must be a positive integer"
  assert M <= X_kn.shape[1], "M must be less than or equal to the number of mel bands (columns) in X_kn"
  K: int = X_kn.shape[0]
  MC_X_ik: np.ndarray = np.zeros((M, K))
  for i in range(1, M + 1):
    for k in range(K):
      tmp = [
        X_kn[k, n - 1] * np.cos(i * (n - 0.5) * np.pi / M)
        for n in range(1, M + 1)
      ]
      MC_X_ik[i - 1, k] = np.sum(tmp)
  return MC_X_ik


def test_large_valid_input():
  X_kn = np.random.rand(10, 5)  # Random values for larger input
  M = 5
  result = get_MC_X_ik(X_kn, M)
  assert result.shape == (
    M, X_kn.shape[0]), f"Expected output shape {(M, X_kn.shape[0])}, but got {result.shape}."
  # Compare with the paper's implementation
  assert np.allclose(result, get_MC_X_ik_from_paper(X_kn, M))


def test_zero():
  X_kn = np.zeros((10, 5))
  M = 5
  result = get_MC_X_ik(X_kn, M)
  assert np.all(result == 0), f"Expected all zeros, but got {result}."
  # Compare with the paper's implementation
  assert np.allclose(result, get_MC_X_ik_from_paper(X_kn, M))


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
  # Compare with the paper's implementation
  assert np.allclose(result, get_MC_X_ik_from_paper(X_kn, M))


def test_basic_input_two_dim():
  X_kn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  M = 2
  expected = np.array([
    [-7.07106781e-01, -7.07106781e-01, -7.07106781e-01],
    [-3.06161700e-16, -6.73555740e-16, -1.04094978e-15]
  ])
  result = get_MC_X_ik(X_kn, M)
  assert np.allclose(result, expected), f"Expected {expected}, but got {result}."
  # Compare with the paper's implementation
  assert np.allclose(result, get_MC_X_ik_from_paper(X_kn, M))


def test_basic_input_one_dim():
  X_kn = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  M = 1
  expected = np.array([
    [6.1232340e-17, 2.4492936e-16, 4.2862638e-16]
  ])
  result = get_MC_X_ik(X_kn, M)
  assert np.allclose(result, expected), f"Expected {expected}, but got {result}."
  # Compare with the paper's implementation
  assert np.allclose(result, get_MC_X_ik_from_paper(X_kn, M))
