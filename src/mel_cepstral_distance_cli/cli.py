import argparse
import logging
import platform
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, List

from mel_cepstral_distance_cli.argparse_helper import get_optional, parse_path
from mel_cepstral_distance_cli.calc_from_mel import init_from_mel_batch_parser, init_from_mel_parser
from mel_cepstral_distance_cli.calc_from_wav import init_from_wav_parser
from mel_cepstral_distance_cli.logging_configuration import (configure_root_logger, get_file_logger,
                                                             init_and_return_loggers,
                                                             try_init_file_buffer_logger)
from mel_cepstral_distance_cli.types import ExecutionResult

__APP_NAME = "mel-cepstral-distance"

__version__ = version(__APP_NAME)

INVOKE_HANDLER_VAR = "invoke_handler"
DEFAULT_LOGGING_BUFFER_CAP = 0


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def get_parsers():
  yield "from-wav", "calculate MCD from two .wav files", init_from_wav_parser
  yield "from-mel", "calculate MCD from two .npy files containing mel-spectrograms", init_from_mel_parser
  yield "from-mel-batch", "calculate MCD from two folders containing mel-spectrograms (.npy)", init_from_mel_batch_parser


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="Command-line interface (CLI) to calculate MCD.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")
  default_log_path = Path(gettempdir()) / f"{__APP_NAME}.log"

  methods = get_parsers()
  for command, description, method in methods:
    method_parser = subparsers.add_parser(
      command, help=description, formatter_class=formatter)
    method_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: method(method_parser),
    })
    logging_group = method_parser.add_argument_group("logging arguments")
    logging_group.add_argument("--log", type=get_optional(parse_path), metavar="FILE",
                               nargs="?", const=None, help="path to write the log", default=default_log_path)
    # logging_group.add_argument("--buffer-capacity", type=parse_positive_integer, default=DEFAULT_LOGGING_BUFFER_CAP,
    #                            metavar="CAPACITY", help="amount of logging lines that should be buffered before they are written to the log-file")
    logging_group.add_argument("--debug", action="store_true",
                               help="include debugging information in log")

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


def parse_args(args: List[str]) -> None:
  configure_root_logger()
  root_logger = getLogger()

  local_debugging = debug_file_exists()
  if local_debugging:
    root_logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit as error:
    error_code = error.args[0]
    # -v -> 0; invalid arg -> 2
    sys.exit(error_code)

  if local_debugging:
    root_logger.debug(f"Parsed arguments: {str(ns)}")

  if not hasattr(ns, INVOKE_HANDLER_VAR):
    parser.print_help()
    sys.exit(0)

  invoke_handler: Callable[..., ExecutionResult] = getattr(ns, INVOKE_HANDLER_VAR)
  delattr(ns, INVOKE_HANDLER_VAR)
  log_to_file = ns.log is not None
  if log_to_file:
    # log_to_file = try_init_file_logger(ns.log, local_debugging or ns.debug)
    log_to_file = try_init_file_buffer_logger(
      ns.log, local_debugging or ns.debug, DEFAULT_LOGGING_BUFFER_CAP)
    if not log_to_file:
      root_logger.warning("Logging to file is not possible.")

  flogger = get_file_logger()
  if not local_debugging:
    sys_version = sys.version.replace('\n', '')
    flogger.debug(f"CLI version: {__version__}")
    flogger.debug(f"Python version: {sys_version}")
    flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

    my_system = platform.uname()
    flogger.debug(f"System: {my_system.system}")
    flogger.debug(f"Node Name: {my_system.node}")
    flogger.debug(f"Release: {my_system.release}")
    flogger.debug(f"Version: {my_system.version}")
    flogger.debug(f"Machine: {my_system.machine}")
    flogger.debug(f"Processor: {my_system.processor}")

  flogger.debug(f"Received arguments: {str(args)}")
  flogger.debug(f"Parsed arguments: {str(ns)}")

  start = perf_counter()
  cmd_flogger, cmd_logger = init_and_return_loggers(__name__)

  # success, changed_anything = invoke_handler(ns, cmd_logger, cmd_flogger)
  try:
    success = invoke_handler(ns, cmd_logger, cmd_flogger)
  except ValueError as error:
    cmd_flogger.debug(error)
    success = False

  exit_code = 0
  if success:
    flogger.info("Everything was successful!")
  else:
    exit_code = 1
    # cmd_logger.error(f"Validation error: {success.default_message}")
    if log_to_file:
      root_logger.error("Not everything was successful! See log for details.")
    else:
      root_logger.error("Not everything was successful!")
    flogger.error("Not everything was successful!")

  duration = perf_counter() - start
  flogger.debug(f"Total duration (s): {duration}")
  if log_to_file and ns.debug:
    # path not encapsulated in "" because it is only console out
    root_logger.info(f"Log: \"{ns.log.absolute()}\"")
  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / f"{__APP_NAME}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{__APP_NAME}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run_prod()
