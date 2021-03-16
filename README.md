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
pipenv run python -m cli print_mcd -a="examples/similar_audios/original.wav" -b="examples/similar_audios/inferred.wav"
```

Output:

```sh
The mel-cepstral distance between the two WAV files is 8.613918026817176. This was computed using 539 frames.
```

This will print a message informing you about the mel-cepstral distance between the audios whose paths were given as arguments and the number of frames that were used in the computation.

There are some parameters with default values which can be specified the following ways:

- n_fft: `-f` or `--n_fft`
- hop_length: `-l` or `--hop_length`
- n_mels: `-m` or `--n_mels`
- n_mfcc: `-c` or `--n_mfcc`

For more information about these parameters, see below.

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
def get_mcd_dtw_from_paths(path_1: str, path_2: str, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 20, n_mfcc: int = 16) -> Tuple[float, int]:
  ...
def get_mcd_dtw_from_mel_spectograms(mel_spectogram_1: np.ndarray, mel_spectogram_2: np.ndarray, n_mfcc: int = 16) -> Tuple[float, int]:
  ...
def get_audio_and_sampling_rate_from_path(path: str) -> Tuple[np.ndarray, int]:
  ...
def get_spectogram(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
  ...
def get_mel_spectogram(spectogram: np.ndarray, sr: int = 22050, n_mels: int = 20) -> np.ndarray:
  ...
def get_mfccs(mel_spectogram: np.ndarray, n_mfcc: int = 16) -> np.ndarray:
  ...
def mel_cepstral_dist_dtw(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  ...
def mel_cepstral_dist(mfccs_1: np.ndarray, mfccs_2: np.ndarray) -> Tuple[float, int]:
  ...
```

If you want to compute the mel-cepstral distance between two WAV files, use `get_mcd_dtw_from_paths` with the paths of the respective audio files as input like in this example:

```py
from mcd import get_mcd_dtw_from_paths

mcd, number_of_frames = get_mcd_dtw_from_paths("examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
```

You can also change the default parameters, those are the following:

- `n_fft`: `n_fft/2+1` is the number of rows of the spectograms and mel-spectograms. It should be a power of two to optimize the speed oft the Fast Fourier Transformation
- `hop_length`: specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore plays a role in computing the spectogram
- `n_mels`: the number of mel bands
- `n_mfcc`: the number of mel-cepstral coefficients per frame that are used to compute the mel-cepstral distance (please notice the zeroth coefficient is not included in the computation, as it is primarily affected by system gain rather than system distortion according to the paper).

The default values for `n_mels` (`20`) and for `n_mfcc` (`16`) are taken from the mentioned paper, the ones for `n_fft` (`1024`) and `hop_length` (`256`) were chosen by me.

There are some parameters for the calculation of the Short Time Fourier Transformation (STFT, used for computing the spectograms) and the mel-filter banks which cannot be set by you when calling the methods. These are:

- `center = False` and `window = 'hamming'` for STFT, see [here](https://librosa.org/doc/latest/generated/librosa.stft.html) to find out more about these parameters. The Hamming-Window was chosen because it was used in a [paper](https://ieeexplore.ieee.org/document/1163420) referenced by Kubichek.
- `norm = None`, `dtype = np.float64`, `htk = True` for the mel-filter banks, see [here](https://librosa.org/doc/latest/generated/librosa.filters.mel.html) for details.

If you want to change any of these parameters, checkout this repository as described above and adjust them where `librosa.stft` and `librosa.filters.mel` are called.

Usually, the number of frames of two different audios does not coincide, but this needs to be the case to compute the mel-cepstral distance. Therefore I use Dynamic Time Warping (dtw) to align the coefficient arrays for both audios (this also enhances comparability). If you want to use a different way to align the arrays, feel free to do so, you can still use this code:

- turn the path of you WAV file into an audio array and the audio's sampling rate with `get_audio_and_sampling_rate_from_path`
- input the audio array in `get_spectogram` and receive a spectogram
- turn the spectogram into a mel-spectogram with `get_mel_spectogram`
- compute the mel-cepstral coefficients with `get_mfccs`, where you use the mel-spectogram as an input
- do the same for the second WAV file
- make sure both mel-cepstral coefficient arrays have the same shape
- input those arrays in `mel_cepstral_dist`

The output values of `get_mcd_dtw_from_paths`, `get_mcd_dtw_from_mel_spectograms`, `mel_cepstral_dist_dtw` and `mel_cepstral_dist` are the mel-cepstral distance and the number of frames per audio (the number of columns of each (aligned) mel-cepstral coefficient array).

## Contributing

If you notice an error, please don't hesitate to open an issue.
