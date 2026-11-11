"""One-ahead prefetch: load item n+1 on a worker thread while the caller processes item n."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar
T = TypeVar("T"); R = TypeVar("R")

def one_ahead(items: Iterable[T], load: Callable[[T], R]) -> Iterator[tuple[T, R]]:
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(load, items[0])
        for i, item in enumerate(items):
            loaded = fut.result()
            if i + 1 < len(items):
                fut = pool.submit(load, items[i + 1])
            yield item, loaded
