import argparse
import logging
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from typing import Callable, List

from mel_cepstral_distance_cli.main import init_mcd_parser

__version__ = version("mel-cepstral-distance")

INVOKE_HANDLER_VAR = "invoke_handler"


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def _init_parser():
  main_parser = ArgumentParser(formatter_class=formatter)
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  method = init_mcd_parser(main_parser)
  main_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: method,
  })

  return main_parser


def configure_logger(productive: bool) -> None:
  loglevel = logging.INFO if productive else logging.DEBUG
  main_logger = getLogger()
  main_logger.setLevel(loglevel)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  logging_formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
    '%Y/%m/%d %H:%M:%S',
  )
  console.setFormatter(logging_formatter)
  console.setLevel(loglevel)


def parse_args(args: List[str], productive: bool = False):
  configure_logger(productive)
  logger = getLogger(__name__)
  logger.debug("Received args:")
  logger.debug(args)
  parser = _init_parser()
  if len(args) == 0:
    parser.print_help()
    return

  received_args = parser.parse_args(args)
  params = vars(received_args)

  if INVOKE_HANDLER_VAR in params:
    invoke_handler: Callable[[ArgumentParser], None] = params.pop(INVOKE_HANDLER_VAR)
    invoke_handler(received_args)
  else:
    parser.print_help()


def run(productive: bool):
  arguments = sys.argv[1:]
  parse_args(arguments, productive)


def run_prod():
  run(True)


if __name__ == "__main__":
  run(not __debug__)
