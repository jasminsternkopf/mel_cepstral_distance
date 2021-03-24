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
- window: `-w` or `--window`
- center: `-c` or `--center`
- n_mels: `-m` or `--n_mels`
- htk: `-t` or `--htk`
- norm: `-o` or `--norm`
- dtype: `-y` or `--dtype`
- n_mfcc: `-c` or `--n_mfcc`
- use_dtw: `-d` or `--use_dtw`

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
```

If you want to compute the mel-cepstral distance between two WAV files, use `get_mcd_between_wav_files` with the paths of the respective audio files as input like in this example:

```py
from mcd import get_mcd_between_wav_files

mcd, penalty, final_frame_number = get_mcd_between_wav_files("examples/similar_audios/original.wav", "examples/similar_audios/inferred.wav")
```

You can also change the default parameters, those are the following:

- `n_fft`: `n_fft/2+1` is the number of rows of the spectograms and mel-spectograms. `n_fft` should be a power of two to optimize the speed oft the Fast Fourier Transformation
- `hop_length`: specifies the number of audio samples between adjacent Short Term Fourier Transformation-columns, therefore plays a role in computing the (mel-)spectograms which are needed to compute the mel-cepstral coefficients
- window: `-w` or `--window`
- center: if `True`, the audio signal `y` is padded so that the frame `D[:, t]` is centered at `y[t*hop_length]` with `D` being the short-time Fourier transform of `y`, otherwise `D[:, t]` begins at `y[t*hop_length]`
- `n_mels`: the number of mel bands
- htk: is a parameter for creating the mel-filter bank. If `True`, the HTK formula is used instead of Slaney
- norm: determines if and how the mel weights are normalized
- dtype: data type of the output
- `n_mfcc`: the number of mel-cepstral coefficients per frame that are used to compute the mel-cepstral distance (please notice the zeroth coefficient is not included in the computation, as it is primarily affected by system gain rather than system distortion according to the paper).
- use_dtw: `-d` or `--use_dtw`

The default values for `n_mels` (`20`) and for `n_mfcc` (`16`) are taken from the mentioned paper, the ones for `n_fft` (`1024`) and `hop_length` (`256`) were chosen by me.

There are some parameters for the calculation of the Short Time Fourier Transformation (STFT, used for computing the spectograms) and the mel-filter banks which cannot be set by you when calling the methods. These are:

- `center = False` and `window = 'hamming'` for STFT, see [here](https://librosa.org/doc/latest/generated/librosa.stft.html) to find out more about these parameters. The Hamming-Window was chosen because it was used in a [paper](https://ieeexplore.ieee.org/document/1163420) referenced by Kubichek.
- `norm = None`, `dtype = np.float64`, `htk = True` for the mel-filter banks, see [here](https://librosa.org/doc/latest/generated/librosa.filters.mel.html) for details.

Usually, the number of frames of two different audios does not coincide, but this needs to be the case to compute the mel-cepstral distance. Therefore I use Dynamic Time Warping (dtw) to align the coefficient arrays for both audios (this also enhances comparability). If you want to use a different way to align the arrays, feel free to do so, you can still use this code:

- turn the path of you WAV file into an audio array and the audio's sampling rate with `get_audio_and_sampling_rate_from_path`
- input the audio array in `get_spectogram` and receive a spectogram
- turn the spectogram into a mel-spectogram with `get_mel_spectogram`
- compute the mel-cepstral coefficients with `get_mfccs`, where you use the mel-spectogram as an input
- do the same for the second WAV file
- make sure both mel-cepstral coefficient arrays have the same shape
- input those arrays in `mel_cepstral_dist`

The output values of `get_mcd_between_wav_files` and `get_mcd_between_audios` are the mel-cepstral distance and the number of frames per audio (the number of columns of each (aligned) mel-cepstral coefficient array).

## Contributing

If you notice an error, please don't hesitate to open an issue.
