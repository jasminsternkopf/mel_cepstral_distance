# Mel-Cepstral Distance

![Python](https://img.shields.io/github/license/jasminsternkopf/mel_cepstral_distance)
![Python](https://img.shields.io/badge/python-3.8-green.svg)

Computation of the mel-cepstral distance of two WAV files based on the paper ["Mel-Cepstral Distance Measure for Objective Speech Quality Assessment"](https://ieeexplore.ieee.org/document/407206) by Robert F. Kubichek.

## Usage as a Standalone Tool

You need Python 3.8.

Checkout this repository if you want to use the client:

```sh
git clone https://github.com/jasminsternkopf/mel_cepstral_distance.git
cd mel_cepstral_distance
python3.8 -m pip install --user pipenv
python3.8 -m pipenv sync
```

You can use the client for example via

```sh
cd src
pipenv run python -m cli print_mcd -a="../examples/similar_audios/original.wav" -b="../examples/similar_audios/inferred.wav"
```

Output:

```sh
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

For more information about these parameters, see .........

## Usage as a Library

In the destination project run:

```sh
# if not already done:
python3.8 -m pip install --user pipenv

# add reference
python3.8 -m pipenv install -e git+https://github.com/jasminsternkopf/mel_cepstral_distance.git@main#egg=mcd
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

All above methods return the mel-cepstral distance, the penalty and the final frame number. Examples and information on the parameters can be found in the ......
Some


## Contributing

If you notice an error, please don't hesitate to open an issue.
