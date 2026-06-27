"""Bounded async seam for blocking Viewer backend work."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from viewer.backend.core.errors import ApiError

ResultT = TypeVar("ResultT")

DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS = 60.0
DEFAULT_BLOCKING_WORK_CONCURRENCY = 4
DEFAULT_BLOCKING_WORK_LIMITER_NAME = "default"
BLOCKING_WORK_TIMEOUT_MESSAGE = (
    "Viewer backend work timed out before it could complete."
)

_blocking_work_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()


def _release_limiter_from_worker(
    loop: asyncio.AbstractEventLoop,
    limiter: asyncio.Semaphore,
) -> None:
    if loop.is_closed():
        return
    loop.call_soon_threadsafe(limiter.release)


async def _await_thread_future(
    future: Future[ResultT],
    *,
    deadline: float,
) -> ResultT:
    loop = asyncio.get_running_loop()
    while not future.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError
        await asyncio.sleep(min(remaining, 0.01))
    return future.result()


def named_blocking_work_limiter(
    name: str,
    concurrency: int,
) -> asyncio.Semaphore:
    """Return a loop-local limiter for a named class of blocking work."""

    if not name:
        raise ValueError("Blocking work limiter name must be non-empty")
    if concurrency < 1:
        raise ValueError("Blocking work limiter concurrency must be at least 1")
    loop = asyncio.get_running_loop()
    limiters = _blocking_work_limiters.get(loop)
    if limiters is None:
        limiters = {}
        _blocking_work_limiters[loop] = limiters
    limiter = limiters.get(name)
    if limiter is None:
        limiter = asyncio.Semaphore(concurrency)
        limiters[name] = limiter
    return limiter


def _default_blocking_work_limiter() -> asyncio.Semaphore:
    return named_blocking_work_limiter(
        DEFAULT_BLOCKING_WORK_LIMITER_NAME,
        DEFAULT_BLOCKING_WORK_CONCURRENCY,
    )


async def run_blocking_io(
    callable_object: Callable[..., ResultT],
    *args: Any,
    timeout_seconds: float = DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
    limiter: asyncio.Semaphore | None = None,
    **kwargs: Any,
) -> ResultT:
    """Run blocking work without tying up the event loop."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    call = partial(callable_object, *args, **kwargs)
    work_limiter = limiter if limiter is not None else _default_blocking_work_limiter()
    acquired = False
    executor: ThreadPoolExecutor | None = None
    future: Future[ResultT] | None = None
    release_when_done = False
    try:
        await asyncio.wait_for(work_limiter.acquire(), timeout_seconds)
        acquired = True
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="viewer-blocking",
        )
        future = executor.submit(call)
        return await _await_thread_future(future, deadline=deadline)
    except TimeoutError as exc:
        if future is not None and not future.done():
            release_when_done = True
            future.add_done_callback(
                lambda _future: _release_limiter_from_worker(loop, work_limiter)
            )
            future.cancel()
        elif future is not None:
            future.cancel()
        raise ApiError(
            BLOCKING_WORK_TIMEOUT_MESSAGE,
            status_code=503,
        ) from exc
    except asyncio.CancelledError:
        if future is not None and not future.done():
            release_when_done = True
            future.add_done_callback(
                lambda _future: _release_limiter_from_worker(loop, work_limiter)
            )
            future.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(
                wait=not release_when_done,
                cancel_futures=release_when_done,
            )
        if acquired:
            if not release_when_done:
                work_limiter.release()
