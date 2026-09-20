"""Bounded process-local execution. A timeout never claims to stop external work."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import BoundedSemaphore
from core.config import settings

_pool = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_TASKS, thread_name_prefix="agent")
_slots = BoundedSemaphore(settings.MAX_CONCURRENT_TASKS)


def submit_bounded(fn):
    if not _slots.acquire(blocking=False):
        raise RuntimeError("Execution capacity is exhausted; retry later.")
    try:
        future = _pool.submit(fn)
    except BaseException:
        _slots.release()
        raise
    future.add_done_callback(lambda _: _slots.release())
    return future


def run_with_timeout(fn, timeout_seconds):
    future = submit_bounded(fn)
    try:
        return future.result(timeout=max(0, timeout_seconds))
    except FutureTimeout:
        future.cancel()
        raise TimeoutError("Execution deadline exceeded. External work may still be running.") from None
