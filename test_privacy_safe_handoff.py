from privacy_safe_handoff import retry_delay


class RetryResponse:
    headers = {"retry-after": "3"}


class RateLimitLike:
    response = RetryResponse()


def test_retry_delay_honors_retry_after() -> None:
    assert retry_delay(RateLimitLike(), 1) == 3.0
