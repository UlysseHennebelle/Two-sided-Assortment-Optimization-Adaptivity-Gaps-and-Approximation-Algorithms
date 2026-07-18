"""Shared script bootstrap and bounded-instance selection."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsao.generation.section7 import GeneratedInstance  # noqa: E402


def bounded_instances(
    instances: Iterable[GeneratedInstance],
    max_size: int | None,
    max_instances: int | None,
) -> Iterator[GeneratedInstance]:
    count = 0
    for generated in instances:
        size = max(generated.instance.num_customers, generated.instance.num_suppliers)
        if max_size is not None and size > max_size:
            continue
        if max_instances is not None and count >= max_instances:
            break
        count += 1
        yield generated
