import logging
import os
from logging import Formatter, Handler, Logger, StreamHandler, getLogger
from logging.handlers import MemoryHandler
from pathlib import Path
from typing import Tuple


class ConsoleFormatter(logging.Formatter):
  """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

  purple = '\x1b[34m'
  blue = '\x1b[36m'
  # blue = '\x1b[38;5;39m'
  yellow = '\x1b[38;5;226m'
  red = '\x1b[1;49;31m'
  bold_red = '\x1b[1;49;31m'
  reset = '\x1b[0m'

  def __init__(self):
    super().__init__()
    self.datefmt = '%H:%M:%S'
    fmt = '(%(levelname)s) %(message)s'
    fmt_info = '%(message)s'

    self.fmts = {
        logging.NOTSET: self.purple + fmt + self.reset,
        logging.DEBUG: self.blue + fmt + self.reset,
        logging.INFO: fmt_info,
        logging.WARNING: self.yellow + fmt + self.reset,
        logging.ERROR: self.red + fmt + self.reset,
        logging.CRITICAL: self.bold_red + fmt + self.reset,
    }

  def format(self, record):
    log_fmt = self.fmts.get(record.levelno)
    formatter = logging.Formatter(log_fmt, self.datefmt)

    return formatter.format(record)


def add_console_out(logger: Logger):
  console = StreamHandler()
  logger.addHandler(console)
  set_console_formatter(console)


def init_and_return_loggers(name: str) -> Tuple[Logger, Logger]:
  logger = getLogger(name)
  flogger = get_file_logger()
  logger.parent = flogger
  logger.handlers.clear()
  assert len(logger.handlers) == 0
  add_console_out(logger)
  return flogger, logger


def set_console_formatter(handler: Handler) -> None:
  logging_formatter = ConsoleFormatter()
  handler.setFormatter(logging_formatter)


def set_logfile_formatter(handler: Handler) -> None:
  fmt = '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s'
  datefmt = '%Y/%m/%d %H:%M:%S'
  logging_formatter = Formatter(fmt, datefmt)
  handler.setFormatter(logging_formatter)


def configure_root_logger() -> None:
  # productive = False
  # loglevel = logging.INFO if productive else logging.DEBUG
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  set_console_formatter(console)
  console.setLevel(logging.DEBUG)


def get_file_logger() -> Logger:
  logger = getLogger("file-logger")
  if logger.propagate:
    logger.propagate = False
  return logger


def try_init_file_logger(path: Path, debug: bool = False) -> bool:
  if path.is_dir():
    logger = getLogger(__name__)
    logger.error("Logging path is a directory!")
    return False
  flogger = get_file_logger()
  assert len(flogger.handlers) == 0
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
      os.remove(path)
    path.write_text("")
    fh = logging.FileHandler(path)
  except Exception as ex:
    logger = getLogger(__name__)
    logger.error("Logfile couldn't be created!")
    logger.exception(ex)
    return False

  set_logfile_formatter(fh)

  level = logging.DEBUG if debug else logging.INFO
  fh.setLevel(level)
  flogger.addHandler(fh)
  return True


def try_init_file_buffer_logger(path: Path, debug: bool = False, buffer_capacity: int = 1000):
  if path.is_dir():
    logger = getLogger(__name__)
    logger.error("Logging path is a directory!")
    return False
  flogger = get_file_logger()
  assert len(flogger.handlers) == 0
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
      os.remove(path)
    path.write_text("")
    fh = logging.FileHandler(path)
  except Exception as ex:
    logger = getLogger(__name__)
    logger.error("Logfile couldn't be created!")
    logger.exception(ex)
    return False

  set_logfile_formatter(fh)

  level = logging.DEBUG if debug else logging.INFO
  fh.setLevel(level)
  mh = MemoryHandler(buffer_capacity, logging.ERROR, fh, True)
  flogger.addHandler(mh)
  return True
