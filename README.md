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
mcd-cli
```

### Example

```sh
# Download two example audio files
wget https://github.com/jasminsternkopf/mel_cepstral_distance/raw/main/examples/similar_audios/original.wav
wget https://github.com/jasminsternkopf/mel_cepstral_distance/raw/main/examples/similar_audios/inferred.wav

# Calculate metrics
mcd-cli original.wav inferred.wav
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
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
```

### Methods

- `get_metrics_wavs`
- `get_metrics_mels`

Both methods return the mel-cepstral distance, the penalty and the final frame number. Examples and information on the parameters can be found in the corresponding documentations.

### Dependencies

- librosa >= 0.9.1
- numpy >= 1.22.3
- scipy >= 1.8.0
- fastdtw >= 0.3.4

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
