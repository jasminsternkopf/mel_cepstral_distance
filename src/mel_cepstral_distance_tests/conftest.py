import logging


def pytest_configure():
  logger = logging.getLogger("numba")
  logger.disabled = True
  logger.propagate = False
