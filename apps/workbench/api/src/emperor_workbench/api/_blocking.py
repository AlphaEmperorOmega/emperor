from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from emperor_workbench.api._container import current_container
from emperor_workbench.api._errors import ApiError

ResultT = TypeVar("ResultT")

DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS = 60.0
DEFAULT_BLOCKING_WORK_CONCURRENCY = 4
DEFAULT_BLOCKING_WORK_LIMITER_NAME = "default"
BLOCKING_WORK_TIMEOUT_MESSAGE = (
    "Workbench backend work timed out before it could complete."
)


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


async def _run_with_executor(
    executor: ThreadPoolExecutor,
    callable_object: Callable[..., ResultT],
    *args: Any,
    timeout_seconds: float,
    limiter: asyncio.Semaphore,
    limiter_already_acquired: bool,
    shutdown_executor: bool,
    **kwargs: Any,
) -> ResultT:
    if limiter_already_acquired and limiter is None:
        raise ValueError("An acquired blocking-work limiter must be provided")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    call = partial(callable_object, *args, **kwargs)
    acquired = limiter_already_acquired
    future: Future[ResultT] | None = None
    release_when_done = False
    try:
        if not acquired:
            await asyncio.wait_for(limiter.acquire(), timeout_seconds)
            acquired = True
        future = executor.submit(call)
        return await _await_thread_future(future, deadline=deadline)
    except TimeoutError as exc:
        if future is not None and not future.done():
            release_when_done = True
            future.add_done_callback(
                lambda _future: _release_limiter_from_worker(loop, limiter)
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
                lambda _future: _release_limiter_from_worker(loop, limiter)
            )
            future.cancel()
        raise
    finally:
        if shutdown_executor:
            executor.shutdown(
                wait=not release_when_done,
                cancel_futures=release_when_done,
            )
        if acquired and not release_when_done:
            limiter.release()


class BlockingWorkRuntime:
    """App-scoped executor and named concurrency limiters for blocking reads."""

    def __init__(self, *, max_workers: int = DEFAULT_BLOCKING_WORK_CONCURRENCY) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="workbench-blocking",
        )
        self._limiters: dict[str, asyncio.Semaphore] = {}
        self._closed = False

    def named_limiter(self, name: str, concurrency: int) -> asyncio.Semaphore:
        if not name:
            raise ValueError("Blocking work limiter name must be non-empty")
        if concurrency < 1:
            raise ValueError("Blocking work limiter concurrency must be at least 1")
        limiter = self._limiters.get(name)
        if limiter is None:
            limiter = asyncio.Semaphore(concurrency)
            self._limiters[name] = limiter
        return limiter

    async def run(
        self,
        callable_object: Callable[..., ResultT],
        *args: Any,
        timeout_seconds: float = DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
        limiter: asyncio.Semaphore | None = None,
        limiter_already_acquired: bool = False,
        **kwargs: Any,
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("Blocking work runtime is closed.")
        work_limiter = limiter or self.named_limiter(
            DEFAULT_BLOCKING_WORK_LIMITER_NAME,
            DEFAULT_BLOCKING_WORK_CONCURRENCY,
        )
        return await _run_with_executor(
            self._executor,
            callable_object,
            *args,
            timeout_seconds=timeout_seconds,
            limiter=work_limiter,
            limiter_already_acquired=limiter_already_acquired,
            shutdown_executor=False,
            **kwargs,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._limiters.clear()


def named_blocking_work_limiter(
    name: str,
    concurrency: int,
) -> asyncio.Semaphore:
    """Return the named limiter owned by the active application."""

    try:
        runtime = current_container().blocking_work
    except RuntimeError:
        if not name:
            raise ValueError("Blocking work limiter name must be non-empty") from None
        if concurrency < 1:
            raise ValueError(
                "Blocking work limiter concurrency must be at least 1"
            ) from None
        return asyncio.Semaphore(concurrency)
    return runtime.named_limiter(name, concurrency)


async def run_blocking_io(
    callable_object: Callable[..., ResultT],
    *args: Any,
    timeout_seconds: float = DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
    limiter: asyncio.Semaphore | None = None,
    limiter_already_acquired: bool = False,
    **kwargs: Any,
) -> ResultT:
    """Run blocking work on the active app runtime or an isolated test runtime."""

    try:
        runtime = current_container().blocking_work
    except RuntimeError:
        work_limiter = limiter or asyncio.Semaphore(DEFAULT_BLOCKING_WORK_CONCURRENCY)
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="workbench-blocking",
        )
        return await _run_with_executor(
            executor,
            callable_object,
            *args,
            timeout_seconds=timeout_seconds,
            limiter=work_limiter,
            limiter_already_acquired=limiter_already_acquired,
            shutdown_executor=True,
            **kwargs,
        )
    return await runtime.run(
        callable_object,
        *args,
        timeout_seconds=timeout_seconds,
        limiter=limiter,
        limiter_already_acquired=limiter_already_acquired,
        **kwargs,
    )


__all__ = [
    "BLOCKING_WORK_TIMEOUT_MESSAGE",
    "BlockingWorkRuntime",
    "named_blocking_work_limiter",
    "run_blocking_io",
]
