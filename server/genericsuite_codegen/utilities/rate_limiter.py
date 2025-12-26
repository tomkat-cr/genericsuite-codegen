
from typing import Optional
import time


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Client identifier.

        Returns:
            bool: True if request is allowed.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []

        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True

        return False

    def get_reset_time(self, client_id: str) -> Optional[float]:
        """
        Get time when rate limit resets for client.

        Args:
            client_id: Client identifier.

        Returns:
            Optional[float]: Reset time as timestamp, None if no limit.
        """
        if client_id not in self.requests or not self.requests[client_id]:
            return None

        oldest_request = min(self.requests[client_id])
        return oldest_request + self.window_seconds
