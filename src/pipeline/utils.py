"""Utilitaires partagés par les étapes du pipeline."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Écrit un JSON déterministe (clés triées) : même contenu ⇒ même hash DVC."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True)


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str) -> int:
    """Nombre de tokens selon l'encodage tiktoken donné (approximation du tokenizer Mistral)."""
    return len(_encoding(encoding_name).encode(text))


def text_hash(text: str) -> str:
    """Clé de cache d'un texte (SHA-256 hexadécimal)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
