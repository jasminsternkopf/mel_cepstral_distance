from argparse import ArgumentParser, Namespace
from typing import Any, Callable

import numpy as np

from mcd.mcd_computation import get_mcd_between_wav_files


def _add_parser_to(subparsers: Any, name: str, init_method: Callable) -> ArgumentParser:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  return parser


def _init_parser() -> ArgumentParser:
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "print_mcd", init_mcd_parser)
  return result


def _process_args(args: Namespace) -> None:
  params = vars(args)
  if "invoke_handler" in params:
    invoke_handler = params.pop("invoke_handler")
    invoke_handler(**params)
  else:
    print("Please specifiy which method you want to invoke.")


def init_mcd_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.add_argument("-a", "--wav_file_1", type=str, required=True,
                      help="Path to first WAV file")
  parser.add_argument("-b", "--wav_file_2", type=str, required=True,
                      help="Path to first WAV file")
  parser.add_argument("-f", "--n_fft", type=int, required=False, default=1024)
  parser.add_argument("-l", "--hop_length", type=int, required=False, default=256)
  parser.add_argument("-w", "--window", type=str, required=False, default="hamming")
  parser.add_argument("-c", "--center", type=bool, required=False, default=False)
  parser.add_argument("-m", "--n_mels", type=int, required=False, default=20)
  parser.add_argument("-t", "--htk", type=bool, required=False, default=True)
  parser.add_argument("-l", "--norm", required=False, default=None)
  parser.add_argument("-y", "--dtype", type=np.dtype, required=False, default=np.float64)
  parser.add_argument("-n", "--no_of_coeffs_per_frame", type=int, required=False, default=16)
  parser.add_argument("-d", "--dtw", type=bool, required=False, default=True)
  return print_mcd_dtw_from_paths


def print_mcd_dtw_from_paths(wav_file_1: str, wav_file_2: str, n_fft: int, hop_length: int, window: str, center: bool, n_mels: int, htk: bool, norm, dtype: np.dtype, no_of_coeffs_per_frame: int, use_dtw: bool):
  mcd, penalty, frames = get_mcd_between_wav_files(
    wav_file_1=wav_file_1,
    wav_file_2=wav_file_2,
    hop_length=hop_length,
    n_fft=n_fft,
    window=window,
    center=center,
    n_mels=n_mels,
    htk=htk,
    norm=norm,
    dtype=dtype,
    no_of_coeffs_per_frame=no_of_coeffs_per_frame,
    use_dtw=use_dtw)
  print(
    f"The mel-cepstral distance between the two WAV files is {mcd} and the penalty is {penalty}. This was computed using {frames} frames.")


if __name__ == "__main__":
  main_parser = _init_parser()
  received_args = main_parser.parse_args()
  _process_args(received_args)
