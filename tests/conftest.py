"""
Disable logging during tests
"""

import logging

import pytest


@pytest.fixture(autouse=True)
def disable_logging():
    """
    Automatically disable logging for all tests.
    Re-enables it after each test.
    """
    logging.disable(logging.CRITICAL)  # Disable all logging
    yield
    logging.disable(logging.NOTSET)  # Re-enable logging
