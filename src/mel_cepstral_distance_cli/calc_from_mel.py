from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd

from mel_cepstral_distance import get_metrics_mels
from mel_cepstral_distance.mcd_computation import get_metrics_mels_pairwise
from mel_cepstral_distance_cli.argparse_helper import (add_dtw_argument, add_n_mfcc_argument,
                                                       parse_existing_directory,
                                                       parse_existing_file, parse_path)
from mel_cepstral_distance_cli.types import ExecutionResult


def init_from_mel_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the Mel-Cepstral Distance and the penalty between two mel files (.npy)."
  parser.add_argument("mel1", type=parse_existing_file, metavar="MEL1",
                      help="path to the first .npy-file")
  parser.add_argument("mel2", type=parse_existing_file, metavar="MEL2",
                      help="path to the second .npy-file")
  add_n_mfcc_argument(parser)
  add_take_log_argument(parser)
  add_dtw_argument(parser)
  return calc_mcd_from_mel_ns


def calc_mcd_from_mel_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  try:
    mel1 = np.load(ns.mel1)
  except Exception as ex:
    logger.error(f"File \"{ns.mel1.absolute()}\" couldn't be loaded!")
    logger.debug(ex)
    return False

  try:
    mel2 = np.load(ns.mel2)
  except Exception as ex:
    logger.error(f"File \"{ns.mel2.absolute()}\" couldn't be loaded!")
    logger.debug(ex)
    return False

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


def init_from_mel_batch_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "This program calculates the Mel-Cepstral Distances and penalties between mel file-pairs (.npy) from two separate folders."
  parser.add_argument("folder1", type=parse_existing_directory, metavar="FOLDER1",
                      help="path to directory containing the first mel-spectrograms, e.g., TTS training data")
  parser.add_argument("folder2", type=parse_existing_directory, metavar="FOLDER2",
                      help="path to directory containing the other mel-spectrograms, e.g., TTS synthesis")
  parser.add_argument("output_csv", type=parse_path, metavar="OUTPUT-CSV",
                      help="path to write the result")
  add_n_mfcc_argument(parser)
  add_take_log_argument(parser)
  add_dtw_argument(parser)
  return calc_mcd_from_mel_batch_ns


def add_take_log_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-t", "--take-log", action="store_true",
                      help="should be set to `False` if log10 already has been applied to the input mel spectrograms, otherwise `True`")


def calc_mcd_from_mel_batch_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  if not cast(Path, ns.output_csv).parent.is_dir():
    logger.error(f"Folder \"{cast(Path, ns.output_csv).parent.absolute()}\" doesn't exist!")
    return False

  df, stats_df, errors_on_files = get_metrics_mels_pairwise(
    ns.folder1, ns.folder2, n_mfcc=ns.n_mfcc, take_log=ns.take_log, use_dtw=ns.dtw, silent=False)

  if len(stats_df) > 0:
    with pd.option_context(
      'display.max_rows', None,
      'display.max_columns', None,
      "display.width", None,
      "display.precision", 4):
      logger.info(f"Statistics:\n{stats_df.to_string(index=False)}")

  all_successful = len(errors_on_files) == 0

  try:
    df.to_csv(ns.output_csv, index=False)
  except ValueError as error:
    logger.debug(error)
    logger.error("Output CSV couldn't be created!")
    return False

  logger.info(f"Written results to: \"{ns.output_csv.absolute()}\"")

  return all_successful
