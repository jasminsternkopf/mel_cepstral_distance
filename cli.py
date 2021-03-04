from argparse import ArgumentParser, Namespace
from typing import Any, Callable

from mcd.mcd_computation import get_mcd_dtw_from_paths


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
  parser.add_argument("-a", "--path_1", type=str, required=True,
                      help="Path to first WAV file")
  parser.add_argument("-b", "--path_2", type=str, required=True,
                      help="Path to first WAV file")
  parser.add_argument("-f", "--n_fft", type=int, required=False, default=1024)
  parser.add_argument("-l", "--hop_length", type=int, required=False, default=256)
  parser.add_argument("-m", "--n_mels", type=int, required=False, default=20)
  parser.add_argument("-c", "--no_of_coeffs_per_frame", type=int, required=False, default=16)
  return print_mcd_dtw_from_paths


def print_mcd_dtw_from_paths(path_1: str, path_2: str, n_fft: int, hop_length: int, n_mels: int, no_of_coeffs_per_frame: int):
  mcd, frames = get_mcd_dtw_from_paths(
    path_1, path_2, n_fft, hop_length, n_mels, no_of_coeffs_per_frame)
  print(
    f"The mel-cepstral distance between the two WAV files is {mcd}. This was computed using {frames} frames.")


if __name__ == "__main__":
  main_parser = _init_parser()
  received_args = main_parser.parse_args()
  _process_args(received_args)
