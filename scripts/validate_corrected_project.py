"""Run the bounded test suite; no medium or large experiment is launched."""

from __future__ import annotations

import sys

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "tests"]))
