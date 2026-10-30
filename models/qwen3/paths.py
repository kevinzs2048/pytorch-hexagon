"""Source paths for the Qwen3 adapter."""
from pathlib import Path


def native_root():
    return Path(__file__).resolve().parent / "native"
