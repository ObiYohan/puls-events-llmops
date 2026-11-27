"""Configuration du projet.

- Hyperparamètres : lus depuis ``params.yaml`` et validés par Pydantic.
  Une clé inconnue, manquante ou hors bornes fait échouer le chargement.
- Secrets : lus depuis les variables d'environnement (et ``.env`` en local),
  sans aucune valeur par défaut.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


class _Section(BaseModel):
    """Base commune : pas de clé inconnue, valeurs immuables."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsParams(_Section):
    raw_events: Path
    ingest_metrics: Path
    chunks_dataset: Path
    chunk_metrics: Path
    embeddings: Path
    embed_metrics: Path
    index_dir: Path
    index_metrics: Path
    reference_set: Path
    eval_metrics: Path
    eval_cases: Path
    system_prompt: Path
    promotion_report: Path

    @field_validator("*")
    @classmethod
    def _resolve_from_root(cls, value: Path) -> Path:
        """Les chemins relatifs sont résolus depuis la racine du projet."""
        return value if value.is_absolute() else PROJECT_ROOT / value


# Champs indispensables aux étapes suivantes (découpage, filtres date et lieu).
REQUIRED_EVENT_FIELDS = frozenset({
    "uid", "longdescription_fr",
    "firstdate_begin", "firstdate_end", "lastdate_begin", "lastdate_end",
})
# Données personnelles des contributeurs OpenAgenda : jamais ingérées.
FORBIDDEN_FIELD_PREFIX = "contributor_"


class IngestParams(_Section):
    base_url: str = Field(pattern=r"^https://")
    region: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    retry_base_seconds: float = Field(ge=0)
    max_days_ahead: int = Field(gt=0)
    fields: list[str] = Field(min_length=1)

    @field_validator("fields")
    @classmethod
    def _check_fields(cls, fields: list[str]) -> list[str]:
        if len(set(fields)) != len(fields):
            raise ValueError("ingest.fields contient des doublons")
        forbidden = [f for f in fields if f.startswith(FORBIDDEN_FIELD_PREFIX)]
        if forbidden:
            raise ValueError(f"champs personnels interdits dans ingest.fields : {forbidden}")
        missing = REQUIRED_EVENT_FIELDS - set(fields)
        if missing:
            raise ValueError(f"champs indispensables absents de ingest.fields : {sorted(missing)}")
        return fields


class ChunkingParams(_Section):
    strip_html: bool
    tokenizer: str
    split_threshold: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    separators: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_overlap(self) -> "ChunkingParams":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        return self


class EmbeddingParams(_Section):
    model: str
    max_tokens_per_batch: int = Field(gt=0)
    pause_seconds: float = Field(ge=0)
    max_retries: int = Field(ge=0)
    retry_base_seconds: float = Field(ge=0)


class RetrievalParams(_Section):
    k: int = Field(ge=1, le=20)
    candidate_factor: int = Field(ge=1)


class GenerationParams(_Section):
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)


class EvaluationThresholds(_Section):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    contextual_precision: float = Field(ge=0, le=1)
    contextual_recall: float = Field(ge=0, le=1)


class EvaluationParams(_Section):
    judge_model: str
    judge_reasoning_effort: Literal["minimal", "low", "medium", "high"]
    max_workers: int = Field(ge=1, le=16)
    max_retries: int = Field(ge=0)
    retry_base_seconds: float = Field(ge=0)
    max_error_rate: float = Field(ge=0, le=1)
    thresholds: EvaluationThresholds


class PromotionParams(_Section):
    tolerance: float = Field(ge=0, le=1)


class MlflowParams(_Section):
    experiment: str = Field(min_length=1)
    registered_model: str = Field(min_length=1)
    champion_alias: str = Field(min_length=1)


class ModelPrice(_Section):
    """Prix en $ par million de tokens."""
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class Params(_Section):
    paths: PathsParams
    ingest: IngestParams
    chunking: ChunkingParams
    embedding: EmbeddingParams
    retrieval: RetrievalParams
    generation: GenerationParams
    evaluation: EvaluationParams
    promotion: PromotionParams
    mlflow: MlflowParams
    pricing: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _check_pricing(self) -> "Params":
        """Chaque modèle utilisé doit avoir un prix : le coût des runs ne doit pas être sous-estimé."""
        used = {self.embedding.model, self.generation.model, self.evaluation.judge_model}
        missing = sorted(used - set(self.pricing))
        if missing:
            raise ValueError(f"prix manquant dans `pricing` pour : {', '.join(missing)}")
        return self


def load_params(path: Path = PARAMS_FILE) -> Params:
    """Charge et valide un fichier de paramètres."""
    with Path(path).open(encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f))


@lru_cache(maxsize=1)
def get_params() -> Params:
    """Paramètres du projet (``params.yaml``), chargés une seule fois."""
    return load_params()


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_secret(name: str) -> str | None:
    """Renvoie la valeur d'un secret, ou ``None`` s'il est absent ou vide.

    Aucune valeur par défaut : à l'appelant de désactiver la fonctionnalité
    concernée (ou d'échouer explicitement) si le secret manque.
    """
    _load_dotenv()
    value = os.getenv(name, "").strip()
    return value or None
