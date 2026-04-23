"""
Recipient Velocity Tracker — Anti-Mule Layer (V4 Step 4)
=========================================================
Implements network-level velocity checks to detect money mule patterns.

If a recipient wallet receives funds from 10+ unique, unlinked senders
within a 60-minute sliding window, it triggers an automatic BLOCK override
— protecting the sender even if their individual profile looks normal.

Architecture:
    In-memory sliding window graph: dict[recipient] -> {senders, timestamps}
    Auto-decays entries older than the window to prevent unbounded growth.
"""
import time
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RecipientWindow:
    """Tracks unique senders and their timestamps for a single recipient."""
    sender_times: dict = field(default_factory=dict)  # sender_hash -> latest_timestamp
    spike_detected: bool = False
    spike_detected_at: float = 0.0


class RecipientVelocityTracker:
    """
    Sliding-window graph analysis for recipient velocity spikes.
    
    Thread-safe: uses a lock for concurrent access from async workers.
    
    Parameters:
        window_seconds: Size of the sliding window (default: 3600 = 60 min)
        sender_threshold: Number of unique senders to trigger a mule alert (default: 10)
        cleanup_interval: How often to prune stale entries (default: 300 = 5 min)
    """

    def __init__(self, window_seconds: int = 3600, sender_threshold: int = 10,
                 cleanup_interval: int = 300):
        self.window_seconds = window_seconds
        self.sender_threshold = sender_threshold
        self.cleanup_interval = cleanup_interval

        self._graph: dict[str, RecipientWindow] = defaultdict(RecipientWindow)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

        # Statistics
        self.total_checks = 0
        self.mule_detections = 0

    def _prune_window(self, window: RecipientWindow, now: float) -> None:
        """Remove sender entries older than the sliding window."""
        cutoff = now - self.window_seconds
        expired = [s for s, t in window.sender_times.items() if t < cutoff]
        for s in expired:
            del window.sender_times[s]
        # Reset spike if all senders have expired
        if not window.sender_times:
            window.spike_detected = False

    def _maybe_cleanup(self, now: float) -> None:
        """Periodically prune the entire graph to prevent memory growth."""
        if now - self._last_cleanup < self.cleanup_interval:
            return
        self._last_cleanup = now
        empty_keys = []
        for recipient, window in self._graph.items():
            self._prune_window(window, now)
            if not window.sender_times:
                empty_keys.append(recipient)
        for k in empty_keys:
            del self._graph[k]

    def check(self, sender_hash: str, recipient_hash: str) -> dict:
        """
        Record a transaction and check if the recipient is exhibiting
        mule-like velocity patterns.

        Returns:
            {
                "mule_detected": bool,
                "unique_senders": int,       # in current window
                "window_seconds": int,
                "reason": str or None,
            }
        """
        now = time.time()
        self.total_checks += 1

        with self._lock:
            self._maybe_cleanup(now)

            window = self._graph[recipient_hash]
            self._prune_window(window, now)

            # Record this sender
            window.sender_times[sender_hash] = now
            unique_count = len(window.sender_times)

            # Check threshold
            is_mule = unique_count >= self.sender_threshold

            if is_mule and not window.spike_detected:
                window.spike_detected = True
                window.spike_detected_at = now
                self.mule_detections += 1
                logger.warning(
                    f"MULE DETECTED: recipient={recipient_hash[:12]}... "
                    f"received from {unique_count} unique senders in {self.window_seconds}s"
                )

            # If previously detected as mule within this window, keep flagging
            if window.spike_detected:
                is_mule = True

        return {
            "mule_detected": is_mule,
            "unique_senders": unique_count,
            "window_seconds": self.window_seconds,
            "reason": (
                f"Recipient received from {unique_count} unique senders "
                f"in {self.window_seconds // 60}min window — mule network suspected"
            ) if is_mule else None,
        }

    def get_stats(self) -> dict:
        """Return tracker statistics for monitoring."""
        with self._lock:
            return {
                "total_checks": self.total_checks,
                "mule_detections": self.mule_detections,
                "tracked_recipients": len(self._graph),
                "window_seconds": self.window_seconds,
                "sender_threshold": self.sender_threshold,
            }

    def get_recipient_status(self, recipient_hash: str) -> Optional[dict]:
        """Check the current velocity status of a specific recipient."""
        with self._lock:
            if recipient_hash not in self._graph:
                return None
            window = self._graph[recipient_hash]
            self._prune_window(window, time.time())
            return {
                "recipient_hash": recipient_hash[:12] + "...",
                "unique_senders": len(window.sender_times),
                "spike_detected": window.spike_detected,
                "threshold": self.sender_threshold,
            }

    def get_recipient_risk_score(self, recipient_hash: str) -> float:
        """
        Dynamic recipient risk score (0.0 - 1.0) based on velocity data.
        Creates a feedback loop: Step 4 anti-mule data → Step 5 model scoring.

        Scale:
            0 senders in window → 0.0
            threshold/2 senders → 0.5
            threshold senders   → 1.0 (capped)
        """
        with self._lock:
            if recipient_hash not in self._graph:
                return 0.0
            window = self._graph[recipient_hash]
            self._prune_window(window, time.time())
            unique = len(window.sender_times)
            # Linear scale: 0 → 0.0, threshold → 1.0
            return min(unique / max(self.sender_threshold, 1), 1.0)
