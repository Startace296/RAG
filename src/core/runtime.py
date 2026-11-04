import os
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout

from dotenv import load_dotenv

from .paths import PROJECT_ROOT


def load_env() -> None:
    """Load settings from the project's ``.env`` file."""
    load_dotenv(PROJECT_ROOT / ".env")


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable written as ``true``/``false``."""
    return os.getenv(name, str(default)).lower() == "true"


def is_verbose() -> bool:
    """Return True when RAG_VERBOSE=true."""
    return env_flag("RAG_VERBOSE")


@contextmanager
def quiet_output() -> Iterator[None]:
    """Silence stdout/stderr (e.g. model download logs) unless RAG_VERBOSE=true."""
    if is_verbose():
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
