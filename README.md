# Mel-Cepstral Distance

[![PyPI](https://img.shields.io/pypi/v/mel-cepstral-distance.svg)](https://pypi.python.org/pypi/mel-cepstral-distance)
[![PyPI](https://img.shields.io/pypi/pyversions/mel-cepstral-distance.svg)](https://pypi.python.org/pypi/mel-cepstral-distance)
[![MIT](https://img.shields.io/github/license/jasminsternkopf/mel_cepstral_distance.svg)](LICENSE)

CLI and library to compute the mel-cepstral distance of two WAV files based on the paper ["Mel-Cepstral Distance Measure for Objective Speech Quality Assessment"](https://ieeexplore.ieee.org/document/407206) by Robert F. Kubichek.

## Installation

```sh
pip install mel-cepstral-distance --user
```

## Usage as CLI

```sh
mcd-cli \
  "../examples/similar_audios/original.wav" \
  "../examples/similar_audios/inferred.wav"
```

Output:

```text
The mel-cepstral distance between the two WAV files is 8.613918026817169 and the penalty is 0.18923933209647492. This was computed using 539 frames.
```

This will print a message informing you about the mel-cepstral distance and penalty between the audios whose paths were given as arguments and the number of frames that were used in the computation.

There are some parameters with default values which can be specified the following ways:

- n_fft: `-f` or `--n_fft`
- hop_length: `-l` or `--hop_length`
- window: `-w` or `--window`
- center: `-c` or `--center`
- n_mels: `-m` or `--n_mels`
- htk: `-t` or `--htk`
- norm: `-o` or `--norm`
- dtype: `-y` or `--dtype`
- n_mfcc: `-c` or `--n_mfcc`
- use_dtw: `-d` or `--use_dtw`

For more information about these parameters, have a look into the documentation of the method `get_mcd_between_wav_files`.

## Usage as a library

```py
from mel_cepstral_distance import get_mcd_between_wav_files, get_mcd_between_audios, get_mcd_between_mel_spectograms
```

### Methods

```py
def get_mcd_between_wav_files(wav_file_1: str, wav_file_2: str, hop_length: int = 256, n_fft: int = 1024,
                              window: str = 'hamming', center: bool = False, n_mels: int = 20, htk: bool = True,
                              norm=None, dtype=np.float64, n_mfcc: int = 16, use_dtw: bool = True
                              ) -> Tuple[float, float, int]:
  ...
def get_mcd_between_audios(audio_1: np.ndarray, audio_2: np.ndarray, sr_1: int, sr_2: int, hop_length: int = 256,
                           n_fft: int = 1024, window: str = 'hamming', center: bool = False, n_mels: int = 20,
                           htk: bool = True, norm=None, dtype=np.float64, n_mfcc: int = 16,
                           use_dtw: bool = True) -> Tuple[float, float, int]:
  ...
def get_mcd_between_mel_spectograms(mel_1: np.ndarray, mel_2: np.ndarray, n_mfcc: int = 16, take_log: bool = True,
                            use_dtw: bool = True) -> Tuple[float, float, int]:
  ...
```

All above methods return the mel-cepstral distance, the penalty and the final frame number. Examples and information on the parameters can be found in the corresponding documentations.

## Contributing

If you notice an error, please don't hesitate to open an issue.

## Citation

If you want to cite this repo, you can use this BibTeX-entry:

```bibtex
@misc{stmcd22,
  author = {Sternkopf, Jasmin and Taubert, Stefan},
  title = {mel-cepstral-distance},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jasminsternkopf/mel_cepstral_distance}}
}
```
