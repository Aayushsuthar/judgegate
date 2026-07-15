import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from types import TracebackType
from typing import Self

from judgegate.errors import ConfigError


def response_key(
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
    salt: str = "",
) -> str:
    """Deterministic content hash identifying one judge call."""
    payload = json.dumps(
        {
            "endpoint": endpoint,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "salt": salt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """SQLite cache so identical judge calls are never paid for twice.

    Probes and reruns hit the same items repeatedly; the cache makes a
    full verify plus probe battery cost roughly one pass over the data.
    A salt separates deliberate reruns (stability probes) from replays.
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        try:
            resolved = Path(path).resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(str(resolved), check_same_thread=False)
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            self._connection.commit()
        except (sqlite3.Error, OSError) as exc:
            raise ConfigError(
                f"could not open the response cache at {path} "
                f"(cache.path in the config): {exc}"
            ) from exc

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return str(row[0]) if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, value)
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
