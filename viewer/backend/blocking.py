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
BLOCKING_WORK_TIMEOUT_MESSAGE = (
    "Viewer backend work timed out before it could complete."
)

_blocking_work_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Semaphore,
] = weakref.WeakKeyDictionary()


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


def _default_blocking_work_limiter() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limiter = _blocking_work_limiters.get(loop)
    if limiter is None:
        limiter = asyncio.Semaphore(DEFAULT_BLOCKING_WORK_CONCURRENCY)
        _blocking_work_limiters[loop] = limiter
    return limiter


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
    timed_out = False
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
        timed_out = True
        raise ApiError(
            BLOCKING_WORK_TIMEOUT_MESSAGE,
            status_code=503,
        ) from exc
    finally:
        if future is not None and not future.done():
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)
        if acquired:
            work_limiter.release()
