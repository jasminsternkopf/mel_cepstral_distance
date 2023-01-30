from argparse import ArgumentParser, Namespace
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from mel_cepstral_distance import get_metrics_mels
from mel_cepstral_distance_cli.argparse_helper import (parse_existing_directory,
                                                       parse_existing_file, parse_path,
                                                       parse_positive_integer)
from mel_cepstral_distance_cli.helper import get_all_files_in_all_subfolders
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
  parser.description = "This program calculates the Mel-Cepstral Distance and the penalty between two mel files (.npy). Both audio files need to have the same sampling rate."
  parser.add_argument("folder1", type=parse_existing_directory, metavar="FOLDER1",
                      help="path to directory containing the first mel-spectrograms")
  parser.add_argument("folder2", type=parse_existing_directory, metavar="FOLDER2",
                      help="path to directory containing the second mel-spectrograms")
  parser.add_argument("output_csv", type=parse_path, metavar="OUTPUT-CSV",
                      help="path to write the result")
  parser.add_argument("-n", "--n-mfcc", type=parse_positive_integer, metavar="N-MFCC", default=16,
                      help="the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion according to Robert F. Kubichek)")
  parser.add_argument("-t", "--take-log", action="store_true",
                      help="should be set to `False` if log10 already has been applied to the input mel spectrograms, otherwise `True`")
  parser.add_argument("-d", "--dtw", action="store_true", help="to compute the mel-cepstral distance, the number of frames has to be the same for both audios; if the parameter is specified, Dynamic Time Warping (DTW) is used to align both arrays containing the respective mel-cepstral coefficients, otherwise the array with less columns is filled with zeros from the right side.")
  return calc_mcd_from_mel_batch_ns


def calc_mcd_from_mel_batch_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:

  all_files = get_all_files_in_all_subfolders(ns.folder1)
  all_wav_files = list(file for file in all_files if file.suffix.lower() == ".npy")

  results: List[Dict[str, Any]] = []
  all_successful = True
  for npy_file_1 in tqdm(all_wav_files, desc="Calculating", unit=" mel(s)"):
    npy_file_2: Path = ns.folder2 / npy_file_1.relative_to(ns.folder1)
    if not npy_file_2.is_file():
      flogger.error(
        f"No matching file for \"{npy_file_1.absolute()}\" at \"{npy_file_2.absolute()}\" found! Skipped.")
      all_successful = False
      continue

    try:
      mel1 = np.load(npy_file_1)
    except Exception as ex:
      logger.error(f"File \"{npy_file_1.absolute()}\" couldn't be loaded!")
      logger.debug(ex)
      all_successful = False
      continue

    try:
      mel2 = np.load(npy_file_2)
    except Exception as ex:
      logger.error(f"File \"{npy_file_2.absolute()}\" couldn't be loaded!")
      logger.debug(ex)
      all_successful = False
      continue

    mcd, penalty, frames = get_metrics_mels(
      mel1,
      mel2,
      n_mfcc=ns.n_mfcc,
      take_log=ns.take_log,
      use_dtw=ns.dtw,
    )

    assert mel1.shape[0] == mel2.shape[0]

    results.append({
      "MEL1": npy_file_1.absolute(),
      "MEL2": npy_file_2.absolute(),
      "N-MFCCs": ns.n_mfcc,
      "Take log": ns.take_log,
      "Use DTW": ns.dtw,
      "# Mel bands": mel1.shape[0],
      "# Frames MEL1": mel1.shape[1],
      "# Frames MEL2": mel2.shape[1],
      "# Frames": frames,
      "MCD": mcd,
      "Penalty": penalty,
    })

  if len(results) == 0:
    logger.addFilter("No files found!")
    return True

  median_mcd = np.median([result["MCD"] for result in results])
  min_mcd = min([result["MCD"] for result in results])
  max_mcd = max([result["MCD"] for result in results])
  median_pen = np.median([result["Penalty"] for result in results])

  logger.info(f"Count of file-pairs: {len(results)}")
  logger.info(f"MCD Min: {min_mcd}")
  logger.info(f"MCD Median: {median_mcd}")
  logger.info(f"MCD Max: {max_mcd}")
  logger.info(f"PEN Median: {median_pen}")

  df = DataFrame(
    data=[result.values() for result in results],
    columns=results[0].keys(),
  )

  try:
    df.to_csv(ns.output_csv, index=False)
  except ValueError as error:
    logger.debug(error)
    logger.error("Output CSV couldn't be created!")
    return False

  logger.info(f"Written results to: \"{ns.output_csv.absolute()}\"")

  return all_successful
