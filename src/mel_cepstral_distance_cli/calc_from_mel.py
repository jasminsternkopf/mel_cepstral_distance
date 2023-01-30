from argparse import ArgumentParser, Namespace
from logging import Logger
from typing import Callable

import numpy as np

from mel_cepstral_distance import get_metrics_mels
from mel_cepstral_distance_cli.argparse_helper import parse_existing_file, parse_positive_integer
from mel_cepstral_distance_cli.types import ExecutionResult


def init_from_mel_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the Mel-Cepstral Distance and the penalty between two mel files (.npy). Both audio files need to have the same sampling rate."
  parser.add_argument("mel1", type=parse_existing_file, metavar="MEL1",
                      help="path to the first .npy-file")
  parser.add_argument("mel2", type=parse_existing_file, metavar="MEL2",
                      help="path to the second .npy-file")
  parser.add_argument("-n", "--n-mfcc", type=parse_positive_integer, metavar="N-MFCC", default=16,
                      help="the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion according to Robert F. Kubichek)")
  parser.add_argument("-t", "--take-log", action="store_true",
                      help="should be set to `False` if log10 already has been applied to the input mel spectrograms, otherwise `True`")
  parser.add_argument("-d", "--dtw", action="store_true", help="to compute the mel-cepstral distance, the number of frames has to be the same for both audios; if the parameter is specified, Dynamic Time Warping (DTW) is used to align both arrays containing the respective mel-cepstral coefficients, otherwise the array with less columns is filled with zeros from the right side.")
  return calc_mcd_from_mel_ns


def calc_mcd_from_mel_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    mel1 = np.load(ns.mel1)
  except Exception as ex:
    raise ValueError("Parameter 'MEL1': File couldn't be loaded!") from ex

  try:
    mel2 = np.load(ns.mel2)
  except Exception as ex:
    raise ValueError("Parameter 'MEL2': File couldn't be loaded!") from ex

  mcd, penalty, frames = get_metrics_mels(
    mel1,
    mel2,
    n_mfcc=ns.n_mfcc,
    take_log=ns.take_log,
    use_dtw=ns.dtw,
  )

  logger.info(f"Mel-Cepstral Distance: {mcd}")
  logger.info(f"Penalty: {penalty}")
  logger.info(f"# Frames: {frames}")
  return True
