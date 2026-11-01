"""Explicit process composition; independent of inference provider and credentials."""
import os


def cloud_enabled(environment=None):
    value = (os.environ if environment is None else environment).get("SEMANTIER_RUNTIME", "0")
    if value not in ("0", "1"):
        raise ValueError("SEMANTIER_RUNTIME must be exactly 0 or 1")
    return value == "1"
