"""Safe loading and identity for repository-owned prompt assets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath


_PROMPT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromptAsset:
    relative_name: str
    text: str
    sha256: str


def load_repo_prompt(relative_name: str, *, allowed_namespace: str) -> PromptAsset:
    path = PurePosixPath(relative_name)
    if path.is_absolute() or path.suffix != ".md" or ".." in path.parts:
        raise ValueError("prompt path must be a relative Markdown asset")
    if not path.parts or path.parts[0] != allowed_namespace:
        raise ValueError("prompt path is outside the allowed namespace")
    namespace_root = (_PROMPT_ROOT / allowed_namespace).resolve()
    resolved = (_PROMPT_ROOT / Path(*path.parts)).resolve()
    if resolved != namespace_root and namespace_root not in resolved.parents:
        raise ValueError("prompt path escapes the allowed namespace")
    text = resolved.read_text(encoding="utf-8")
    return PromptAsset(relative_name=path.as_posix(), text=text, sha256=sha256(text.encode()).hexdigest())
