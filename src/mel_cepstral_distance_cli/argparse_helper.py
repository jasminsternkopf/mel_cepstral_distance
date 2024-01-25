import argparse
import codecs
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


class ConvertToSetAction(argparse._StoreAction):
  def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Optional[List], option_string: Optional[str] = None):
    if values is not None:
      values = set(values)
    super().__call__(parser, namespace, values, option_string)


def add_dtw_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-d", "--dtw", action="store_true", help="to compute the mel-cepstral distance, the number of frames has to be the same for both audios; if the parameter is specified, Dynamic Time Warping (DTW) is used to align both arrays containing the respective mel-cepstral coefficients, otherwise the array with less columns is filled with zeros from the right side.")


def add_n_mfcc_argument(parser: ArgumentParser) -> None:
  parser.add_argument("-n", "--n-mfcc", type=parse_positive_integer, metavar="N-MFCC", default=16,
                      help="the number of mel-cepstral coefficients that are computed per frame, starting with the first coefficient (the zeroth coefficient is omitted, as it is primarily affected by system gain rather than system distortion according to Robert F. Kubichek)")


def parse_codec(value: str) -> str:
  value = parse_required(value)
  try:
    codecs.lookup(value)
  except LookupError as error:
    raise ArgumentTypeError("Codec was not found!") from error
  return value


def parse_path(value: str) -> Path:
  value = parse_required(value)
  try:
    path = Path(value)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a path!") from error
  return path


def parse_optional_value(value: str, method: Callable[[str], T]) -> Optional[T]:
  if value is None:
    return None
  return method(value)


def get_optional(method: Callable[[str], T]) -> Callable[[str], Optional[T]]:
  result = partial(
    parse_optional_value,
    method=method,
  )
  return result


def parse_existing_file(value: str) -> Path:
  path = parse_path(value)
  if not path.is_file():
    raise ArgumentTypeError("File was not found!")
  return path


def parse_existing_directory(value: str) -> Path:
  path = parse_path(value)
  if not path.is_dir():
    raise ArgumentTypeError("Directory was not found!")
  return path


def parse_required(value: Optional[str]) -> str:
  if value is None:
    raise ArgumentTypeError("Value must not be None!")
  return value


def parse_non_empty(value: Optional[str]) -> str:
  value = parse_required(value)
  if value == "":
    raise ArgumentTypeError("Value must not be empty!")
  return value


def parse_non_empty_or_whitespace(value: str) -> str:
  value = parse_required(value)
  if value.strip() == "":
    raise ArgumentTypeError("Value must not be empty or whitespace!")
  return value


def parse_float(value: str) -> float:
  value = parse_required(value)
  try:
    value = float(value)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a decimal number!") from error
  return value


def parse_positive_float(value: str) -> float:
  value = parse_float(value)
  if not value > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return value


def parse_non_negative_float(value: str) -> float:
  value = parse_float(value)
  if not value >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return value


def parse_integer(value: str) -> int:
  value = parse_required(value)
  if not value.isdigit():
    raise ArgumentTypeError("Value needs to be an integer!")
  value = int(value)
  return value


def parse_positive_integer(value: str) -> int:
  value = parse_integer(value)
  if not value > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return value


def parse_integer_greater_one(value: str) -> int:
  value = parse_integer(value)
  if not value > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return value


def parse_float_greater_one(value: str) -> int:
  value = parse_float(value)
  if not value > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return value


def parse_non_negative_integer(value: str) -> int:
  value = parse_integer(value)
  if not value >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return value
