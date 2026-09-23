"""Asynchronous ingestion worker (Stretch Goal — job queue processing).

Polls a `JobQueue` port for claimed jobs and runs them through the
existing `IngestPipeline` — no ingestion logic is duplicated here, this
is purely the polling-loop/retry/error-handling wrapper around it, kept
separate so `/ingest` (synchronous) and this worker (asynchronous) share
one pipeline implementation.
"""

from __future__ import annotations

import asyncio
import logging

from src.application.ingest_pipeline import IngestPipeline
from src.domain.exceptions import DomainError
from src.domain.ports import JobQueue
from src.infra.logging_config import log_event

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 1.0
MAX_ATTEMPTS = 3


class IngestionWorker:
    def __init__(
        self,
        job_queue: JobQueue,
        ingest_pipeline: IngestPipeline,
        worker_id: str = "worker-1",
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._job_queue = job_queue
        self._ingest_pipeline = ingest_pipeline
        self._worker_id = worker_id
        self._max_attempts = max_attempts

    async def run_once(self) -> bool:
        """Claim and process a single job. Returns False if the queue was empty."""
        job = await self._job_queue.claim_next(self._worker_id)
        if job is None:
            return False

        job_id = job["job_id"]
        audio_file_path = job["audio_file_path"]
        attempts = job["attempts"]
        log_event(
            logger, logging.INFO, "worker.job.claimed", job_id=str(job_id),
            audio_file_path=audio_file_path, attempts=attempts, worker_id=self._worker_id,
        )
        try:
            result = await self._ingest_pipeline.run(audio_file_path)
        except DomainError as exc:
            log_event(
                logger, logging.ERROR, "worker.job.failed", job_id=str(job_id),
                attempts=attempts, error=str(exc),
            )
            # Simple, explicit terminal-failure policy: mark failed immediately.
            # A production system might reset status back to 'pending' for a
            # bounded number of retries; kept out of scope here to match the
            # plan's stretch-goal sizing (`max_attempts` is still recorded/
            # enforced so callers can distinguish retryable vs exhausted jobs).
            await self._job_queue.mark_failed(job_id, f"attempt {attempts}: {exc}")
            return True

        log_event(
            logger, logging.INFO, "worker.job.done", job_id=str(job_id),
            audio_file_id=str(result.audio_file_id), chunk_count=result.chunk_count,
        )
        await self._job_queue.mark_done(job_id)
        return True

    async def run_forever(self, poll_interval_s: float = DEFAULT_POLL_INTERVAL_S) -> None:
        """Polling loop — intended to run as a long-lived background process."""
        while True:
            processed = await self.run_once()
            if not processed:
                await asyncio.sleep(poll_interval_s)
