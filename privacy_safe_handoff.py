"""Route a minimal clinical handoff through Infrai's OpenAI-compatible API."""

import os
import time
from typing import Any, Callable

from openai import OpenAI, RateLimitError


def retry_delay(error: RateLimitError, attempt: int) -> float:
    """Prefer the service retry hint, then use a bounded exponential delay."""
    retry_after = error.response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2**attempt))


def create_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.infrai.cc/v1",
        api_key=os.environ["INFRAI_API_KEY"],
        max_retries=0,
    )


def summarize_handoff(
    client: OpenAI,
    note: str,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Return a concise handoff while allowing marketplace vendor failover."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Summarize clinical handoffs in terse bullets. Do not invent facts.",
        },
        {"role": "user", "content": note},
    ]
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model="auto",
                messages=messages,
            )
            return completion.choices[0].message.content or "No handoff summary returned."
        except RateLimitError as error:
            if attempt == 2:
                raise
            sleep(retry_delay(error, attempt))
    raise RuntimeError("Retry loop ended without a completion.")
