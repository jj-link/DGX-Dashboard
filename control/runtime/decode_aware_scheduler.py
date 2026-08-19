"""vLLM scheduler policy that bounds prefill interference with active decode."""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)


class DecodeAwareScheduler(Scheduler):
    """Run prefills on a cadence while any request is decoding.

    vLLM's scheduler already knows how to defer in-progress and waiting prefills
    through ``throttle_prefills``. The base engine only activates that path for
    data-parallel balancing. This subclass applies the same mechanism to TP
    serving so an in-progress prefill cannot occupy every mixed scheduler step.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decode_aware_prefill_interval = int(
            os.environ.get("DECODE_AWARE_PREFILL_INTERVAL", "1")
        )
        if self.decode_aware_prefill_interval < 1:
            raise ValueError("DECODE_AWARE_PREFILL_INTERVAL must be at least 1")
        logger.info(
            "Decode-aware prefill cadence interval: %d",
            self.decode_aware_prefill_interval,
        )

    def schedule(self, throttle_prefills: bool = False):
        next_step = self.current_step + 1
        has_active_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        cadence_throttle = (
            has_active_decode
            and self.decode_aware_prefill_interval > 1
            and next_step % self.decode_aware_prefill_interval != 0
        )
        if cadence_throttle:
            # The base DP policy bypasses throttling when it considers prefill
            # capacity saturated. Interactive TP fairness must remain bounded
            # even with queued prefills, so force this step to stay decode-only.
            self.prefill_capacity_bound = False
        return super().schedule(
            throttle_prefills=throttle_prefills or cadence_throttle
        )
