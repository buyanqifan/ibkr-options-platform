"""Async event bridge: runs ib_insync's asyncio loop in a dedicated thread,
exposing a synchronous interface for Dash callbacks."""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine
from utils.logger import setup_logger

logger = setup_logger("event_bridge")


class AsyncEventBridge:
    """Bridge between Dash's synchronous callbacks and ib_insync's async event loop.

    Starts a dedicated daemon thread running an asyncio event loop.
    Provides ``run_coroutine`` to submit coroutines from any thread and
    block until the result is ready.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background event loop thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ib-event-loop")
        self._thread.start()
        self._started.wait(timeout=10)
        logger.info("AsyncEventBridge started")

    def stop(self):
        """Gracefully shut down the event loop."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AsyncEventBridge stopped")

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Event bridge not started")
        return self._loop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_coroutine(self, coro: Coroutine, timeout: float = 30.0) -> Any:
        """Submit a coroutine to the event loop and block until result is ready.

        Safe to call from any thread (including Dash callback threads).
        Handles "event loop is already running" errors gracefully.
        """
        try:
            future: Future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(f"Coroutine timed out after {timeout}s")
        except Exception as e:
            # Handle "event loop is already running" error
            error_msg = str(e)
            if "event loop is already running" in error_msg or "This event loop is already running" in error_msg:
                logger.warning(f"Event loop conflict detected: {e}")
                # Fallback: run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result(timeout=timeout)
            # Re-raise other exceptions
            raise

    def submit_async(self, coro: Coroutine) -> Future:
        """Submit a coroutine without blocking. Returns a Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            self._loop = None
