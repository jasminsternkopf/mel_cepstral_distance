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

  _add_parser_to(subparsers, "get_mcd_dtw_from_paths", init_mcd_parser)
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
  parser.add_argument("-f", "--n_fft", type=int, required=False)
  parser.add_argument("-h", "--hop_length", type=int, required=False)
  parser.add_argument("-m", "--n_mels", type=int, required=False)
  parser.add_argument("-c", "--no_of_coeffs_per_frame", type=int, required=False)
  return get_mcd_dtw_from_paths


if __name__ == "__main__":
  main_parser = _init_parser()
  received_args = main_parser.parse_args()
  _process_args(received_args)
