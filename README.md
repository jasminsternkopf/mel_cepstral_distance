# Mel-Cepstral Distance

[![PyPI](https://img.shields.io/pypi/v/mel-cepstral-distance.svg)](https://pypi.python.org/pypi/mel-cepstral-distance)
[![PyPI](https://img.shields.io/pypi/pyversions/mel-cepstral-distance.svg)](https://pypi.python.org/pypi/mel-cepstral-distance)
[![MIT](https://img.shields.io/github/license/jasminsternkopf/mel_cepstral_distance.svg)](https://github.com/jasminsternkopf/mel_cepstral_distance/blob/main/LICENSE)
![PyPI](https://img.shields.io/pypi/implementation/mel-cepstral-distance.svg)
[![PyPI](https://img.shields.io/github/commits-since/jasminsternkopf/mel_cepstral_distance/latest/main.svg)](https://github.com/jasminsternkopf/mel_cepstral_distance/compare/v0.0.3...main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10567255.svg)](https://doi.org/10.5281/zenodo.10567255)

CLI and library to compute the mel-cepstral distance of two WAV files based on the paper ["Mel-Cepstral Distance Measure for Objective Speech Quality Assessment"](https://ieeexplore.ieee.org/document/407206) by Robert F. Kubichek.

## Installation

```sh
pip install mel-cepstral-distance --user
```

## Usage as CLI

```sh
mcd-cli
```

### Example

```sh
# Download two example audio files
wget https://github.com/jasminsternkopf/mel_cepstral_distance/raw/main/examples/similar_audios/original.wav
wget https://github.com/jasminsternkopf/mel_cepstral_distance/raw/main/examples/similar_audios/inferred.wav

# Calculate metrics
mcd-cli from-wav original.wav inferred.wav
```

Output:

```text
Mel-Cepstral Distance: 19.013673608495836
Penalty: 0.11946050096339111
# Frames: 519
```

This will print a message informing you about the mel-cepstral distance and penalty between the audios whose paths were given as arguments and the number of frames that were used in the computation.

## Usage as a library

```py
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels, get_metrics_mels_pairwise
```

### Main methods

- `get_metrics_wavs`
- `get_metrics_mels`

Both methods return the mel-cepstral distance, the penalty and the final frame number. Examples and information on the parameters can be found in the corresponding documentations.

## Development setup

```sh
# update
sudo apt update
# install Python 3.8-3.11 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/jasminsternkopf/mel_cepstral_distance.git
cd mel_cepstral_distance
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first, install the tool (see "Development setup")
# then, navigate into the directory of the repo
cd mel_cepstral_distance
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

## License

MIT License

## References

- Kubichek, R. “Mel-Cepstral Distance Measure for Objective Speech Quality Assessment.” In Proceedings of IEEE Pacific Rim Conference on Communications Computers and Signal Processing, 1:125–28. Victoria, BC, Canada: IEEE, 1993. https://doi.org/10.1109/PACRIM.1993.407206.
- Muda, Lindasalwa, Mumtaj Begam, and I. Elamvazuthi. “Voice Recognition Algorithms Using Mel Frequency Cepstral Coefficient (MFCC) and Dynamic Time Warping (DTW) Techniques.” Journal of Computing vol. 2, no. 3 (March 2010): 6.

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 416228727 – CRC 1410

## Citation

If you want to cite this repo, you can use the BibTeX-entry generated by GitHub (see *About => Cite this repository*).

```txt
Sternkopf, J., & Taubert, S. (2024). mel-cepstral-distance (Version 0.0.3) [Computer software]. https://doi.org/10.5281/zenodo.10567255
```

## FAQ

### How were the default parameters set?

We based some of the parameters on the two mentioned references and set the other ones by ourselves depending on the parameter description of the underlying libraries:

- `hop-length` -> 256: Kubichek & Muda et al.
- `window` -> hamming: Muda et al.
- `n-mels` -> 20: Kubichek
  - [Battenberg et al. (2019)](https://arxiv.org/abs/1906.03402) computed the first 13 MFCCs
- `n-mfcc` -> 16: by us
- `n-fft` -> 1024: by us
- `center` -> False: by us
- `htk` -> False: by us
- `norm` -> None: by us
- `dtw` -> True: by us
  - calculate the MCD-DTW, which is used as metric in works like:
    - [Shah et al., 2014](https://ieeexplore.ieee.org/abstract/document/6853600)
    - [VoiceLoop (Taigman et al., 2018)](https://arxiv.org/abs/1707.06588)
    - [Capacitron (Battenberg et al., 2019)](https://arxiv.org/abs/1906.03402)
    - [Attentron (Choi et al., 2020)](https://arxiv.org/abs/2005.08484)

### Why is Python 3.12 not supported?

The dependency `numba` is currently not available for Python 3.12.
