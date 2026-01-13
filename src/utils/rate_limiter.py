"""Rate limit handling utilities with exponential backoff and 429 detection."""

import time
import logging
from typing import Optional, Callable, Any, Dict
from functools import wraps

import requests

logger = logging.getLogger(__name__)

# Global rate limit state per domain
_RATE_LIMIT_STATE: Dict[str, Dict] = {}


class RateLimitError(Exception):
    """Raised when rate limit is hit and all retries exhausted."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


def get_retry_after(response: requests.Response) -> Optional[float]:
    """Extract retry-after time from response headers."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            # Could be HTTP date format, default to 60s
            return 60.0

    # Check common rate limit headers
    reset_at = response.headers.get("X-RateLimit-Reset")
    if reset_at:
        try:
            reset_ts = float(reset_at)
            wait_time = reset_ts - time.time()
            return max(1.0, wait_time)
        except ValueError:
            pass

    return None


def is_rate_limited(response: requests.Response) -> bool:
    """Check if response indicates rate limiting."""
    return response.status_code == 429 or response.status_code == 503


def get_domain(url: str) -> str:
    """Extract domain from URL for rate limit tracking."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc


def check_rate_limit_cooldown(domain: str) -> Optional[float]:
    """Check if domain is in rate limit cooldown. Returns seconds to wait or None."""
    state = _RATE_LIMIT_STATE.get(domain)
    if not state:
        return None

    cooldown_until = state.get("cooldown_until", 0)
    now = time.time()
    if now < cooldown_until:
        return cooldown_until - now
    return None


def set_rate_limit_cooldown(domain: str, seconds: float) -> None:
    """Set rate limit cooldown for a domain."""
    _RATE_LIMIT_STATE[domain] = {
        "cooldown_until": time.time() + seconds,
        "last_hit": time.time(),
    }
    logger.warning(f"Rate limit hit on {domain}, cooling down for {seconds:.1f}s")


def clear_rate_limit_cooldown(domain: str) -> None:
    """Clear rate limit cooldown for a domain."""
    if domain in _RATE_LIMIT_STATE:
        del _RATE_LIMIT_STATE[domain]


def request_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    backoff_multiplier: float = 2.0,
    timeout: float = 10.0,
    **kwargs
) -> requests.Response:
    """Make HTTP request with exponential backoff and rate limit handling.

    Args:
        method: HTTP method (GET, POST, etc)
        url: Request URL
        max_retries: Maximum retry attempts
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_multiplier: Multiplier for exponential backoff
        timeout: Request timeout in seconds
        **kwargs: Additional arguments to pass to requests

    Returns:
        Response object

    Raises:
        RateLimitError: If rate limit exhausted after all retries
        requests.RequestException: For other request errors
    """
    domain = get_domain(url)

    # Check if domain is in cooldown
    cooldown = check_rate_limit_cooldown(domain)
    if cooldown and cooldown > 5:
        raise RateLimitError(f"Domain {domain} is rate limited", retry_after=cooldown)
    elif cooldown:
        time.sleep(cooldown)

    backoff = initial_backoff
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)

            if is_rate_limited(resp):
                retry_after = get_retry_after(resp) or backoff
                set_rate_limit_cooldown(domain, retry_after)

                if attempt < max_retries - 1:
                    wait_time = min(retry_after, max_backoff)
                    logger.info(f"Rate limited on {domain}, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    backoff = min(backoff * backoff_multiplier, max_backoff)
                    continue
                else:
                    raise RateLimitError(f"Rate limit exceeded for {domain}", retry_after=retry_after)

            # Success - clear any cooldown
            clear_rate_limit_cooldown(domain)
            return resp

        except requests.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on {url}, retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * backoff_multiplier, max_backoff)
            else:
                raise

        except requests.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Connection error on {url}, retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * backoff_multiplier, max_backoff)
            else:
                raise

    if last_error:
        raise last_error
    raise requests.RequestException(f"Failed after {max_retries} attempts")


def get_with_retry(url: str, **kwargs) -> requests.Response:
    """Convenience function for GET requests with retry."""
    return request_with_retry("GET", url, **kwargs)


def post_with_retry(url: str, **kwargs) -> requests.Response:
    """Convenience function for POST requests with retry."""
    return request_with_retry("POST", url, **kwargs)
