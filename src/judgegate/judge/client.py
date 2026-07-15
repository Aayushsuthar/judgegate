import contextlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from judgegate.config import JudgeSettings
from judgegate.errors import ConfigError, JudgeError
from judgegate.judge.cache import ResponseCache, response_key

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class JudgeClient:
    """Calls any OpenAI-compatible chat completions endpoint.

    Retries transient failures with exponential backoff, honors
    Retry-After headers, runs batches with bounded concurrency, and
    consults the response cache before spending a single token.
    """

    def __init__(
        self,
        settings: JudgeSettings,
        cache: ResponseCache | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        api_key = os.environ.get(settings.api_key_env, "")
        if not api_key:
            raise ConfigError(
                f"environment variable {settings.api_key_env} is not set; "
                "the judge endpoint needs an API key"
            )
        self._client = httpx.Client(
            base_url=settings.endpoint.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def complete(self, prompt: str, salt: str = "") -> str:
        """One judge response for one prompt, cached when possible."""
        key = response_key(
            self.settings.endpoint,
            self.settings.model,
            self.settings.temperature,
            self.settings.max_tokens,
            prompt,
            salt,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        text = self._request(prompt)
        if self.cache is not None:
            self.cache.put(key, text)
        return text

    def complete_many(self, prompts: dict[str, str], salt: str = "") -> dict[str, str]:
        """Judge responses for many prompts, keyed the same way as the input."""
        if not prompts:
            return {}
        with ThreadPoolExecutor(max_workers=self.settings.concurrency) as pool:
            futures = {
                key: pool.submit(self.complete, prompt, salt)
                for key, prompt in prompts.items()
            }
            return {key: future.result() for key, future in futures.items()}

    def _request(self, prompt: str) -> str:
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        attempts = self.settings.retries + 1
        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc}"
                if attempt < attempts - 1:
                    self._backoff(attempt, None)
                    continue
                break
            if response.status_code == 200:
                return self._parse(response)
            body = response.text[:200]
            last_error = f"HTTP {response.status_code}: {body}"
            if response.status_code in _RETRYABLE_STATUS and attempt < attempts - 1:
                self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            break
        raise JudgeError(
            f"judge call failed after {attempts} attempt(s); last error: {last_error}"
        )

    def _parse(self, response: httpx.Response) -> str:
        try:
            data: Any = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise JudgeError(
                f"judge endpoint returned an unexpected response shape: {response.text[:200]}"
            ) from exc
        if not isinstance(content, str):
            raise JudgeError("judge response content is not text")
        return content

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(2.0**attempt, 30.0)
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        time.sleep(min(delay, 60.0))
