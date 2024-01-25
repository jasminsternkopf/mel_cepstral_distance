import shutil
import tempfile
from pathlib import Path

import numpy as np
from librosa import load
from librosa.feature import melspectrogram

from mel_cepstral_distance.mcd_computation import get_metrics_mels_pairwise
from mel_cepstral_distance_tests.test_get_metrics_wavs import *


def test_empty_dirs():
  dir_a_path = Path(tempfile.mkdtemp())
  dir_b_path = Path(tempfile.mkdtemp())

  # add one fail
  (dir_a_path / '0.npy').write_text("invalid content", encoding="utf8")

  df, stats_df, fails = get_metrics_mels_pairwise(dir_a_path, dir_b_path)

  shutil.rmtree(dir_a_path)
  shutil.rmtree(dir_b_path)

  assert list(df.columns) == ['MEL1', 'MEL2', 'Name', '#MFCCs', 'Take log?',
                              'Use DTW?', '#Mel-bands', '#Frames MEL1', '#Frames MEL2', '#Frames', 'MCD', 'PEN']
  assert len(df) == 0
  assert list(stats_df.columns) == ["Metric", "Min", "Q1",
                                    "Q2", "Q3", "Max", "Mean", "SD", "Kurt", "Skew"]
  assert len(stats_df) == 0
  assert fails == [
    dir_a_path / '0.npy'
  ]


def test_component():
  hop_length: int = 256
  n_fft: int = 1024
  window: str = 'hamming'
  center: bool = False
  n_mels: int = 20
  htk: bool = True
  norm = None
  dtype: np.dtype = np.float64
  pairs = [
    (SIM_ORIG, SIM_INF),
    (SOSIM_ORIG, SOSIM_INF),
    (DISSIM_ORIG, DISSIM_INF),
  ]

  dir_a_path = Path(tempfile.mkdtemp())
  dir_b_path = Path(tempfile.mkdtemp())
  for i, (f1, f2) in enumerate(pairs):
    target1 = dir_a_path / f"{i}.npy"
    target2 = dir_b_path / f"{i}.npy"

    audio_1, sr_1 = load(f1, sr=None, mono=True, res_type=None,
                         offset=0.0, duration=None, dtype=np.float32)
    audio_2, sr_2 = load(f2, sr=None, mono=True, res_type=None,
                         offset=0.0, duration=None, dtype=np.float32)

    mel_spectrogram1 = melspectrogram(
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

    mel_spectrogram2 = melspectrogram(
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

    np.save(target1, mel_spectrogram1)
    np.save(target2, mel_spectrogram2)

  # Fails
  # case 1
  shutil.copy(dir_a_path / '0.npy', dir_a_path / '6.npy')
  # case 2
  (dir_a_path / '4.npy').write_text("invalid content", encoding="utf8")
  shutil.copy(dir_b_path / '0.npy', dir_b_path / '4.npy')
  # case 3
  shutil.copy(dir_a_path / '0.npy', dir_a_path / '5.npy')
  (dir_b_path / '5.npy').write_text("invalid content", encoding="utf8")

  df, stats_df, fails = get_metrics_mels_pairwise(dir_a_path, dir_b_path)

  shutil.rmtree(dir_a_path)
  shutil.rmtree(dir_b_path)

  df_np = df.to_numpy()
  assert_result = np.array([
      [dir_a_path / '0.npy',
       dir_b_path / '0.npy', '0', 16, True, True, 20.0,
       519.0, 457.0, 539.0, 8.613918026817169, 0.18923933209647492],
      [dir_a_path / '1.npy',
       dir_b_path / '1.npy', '1', 16, True, True, 20.0,
       822.0, 835.0, 952.0, 9.62103176737408, 0.259453781512605],
      [dir_a_path / '2.npy',
       dir_b_path / '2.npy', '2', 16, True, True, 20.0,
       822.0, 920.0, 1015.0, 13.983229820898327, 0.283743842364532],
      [dir_a_path,
       dir_b_path, 'ALL', 16, True, True, 20.0,
       822.0, 835.0, 952.0, 9.62103176737408, 0.259453781512605]
    ],
    dtype=object
  )

  assert list(df.columns) == ['MEL1', 'MEL2', 'Name', '#MFCCs', 'Take log?',
                              'Use DTW?', '#Mel-bands', '#Frames MEL1', '#Frames MEL2', '#Frames', 'MCD', 'PEN']
  assert len(df_np) == 4
  for i in range(4):
    np.testing.assert_almost_equal(df_np[i, -2], assert_result[i, -2])
    np.testing.assert_almost_equal(df_np[i, -1], assert_result[i, -1])
    df_np[i, -2] = 0
    df_np[i, -1] = 0
    assert_result[i, -2] = 0
    assert_result[i, -1] = 0
  np.testing.assert_array_equal(df_np, assert_result)

  stats_df_np = stats_df.to_numpy()
  assert_stats_result_np = np.array(
    [
      [
        'MCD', 8.613918026817169, 9.117474897095624, 9.62103176737408,
          11.802130794136204, 13.983229820898327, 10.739393205029858,
          2.8540193612590263, np.nan, 1.4925846186224958
      ],
      [
        'PEN', 0.18923933209647492, 0.22434655680453996,
        0.259453781512605, 0.2715988119385685, 0.283743842364532,
        0.24414565199120397, 0.049076773909195764, np.nan,
        -1.2670807645776279
      ],
    ],
    dtype=object
  )

  assert list(stats_df.columns) == ["Metric", "Min", "Q1",
                                    "Q2", "Q3", "Max", "Mean", "SD", "Kurt", "Skew"]
  np.testing.assert_equal(stats_df_np[:, :1], assert_stats_result_np[:, :1])
  np.testing.assert_almost_equal(stats_df_np[:, 1:].astype(
    float), assert_stats_result_np[:, 1:].astype(float), verbose=True)

  assert fails == [
    dir_a_path / '4.npy',
    dir_b_path / '5.npy',
    dir_a_path / '6.npy',
  ]
