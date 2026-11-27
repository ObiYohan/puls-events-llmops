"""Index servi par l'API : la version « champion » du registre MLflow.

- Si ``MLFLOW_TRACKING_URI`` est définie (conteneur, CI), l'API télécharge
  ``models:/<registered_model>/<version>`` — la version pointée par l'alias
  champion — dans ``paths.served_index_dir``. Aucune donnée n'est copiée dans
  l'image Docker. Si le registre est injoignable, le démarrage échoue
  explicitement plutôt que de servir un index périmé.
- Sinon (développement local), l'API sert l'index produit par ``dvc repro``
  dans ``paths.index_dir``.

Dans les deux cas, le manifeste est vérifié : l'index doit avoir été encodé
avec le modèle d'embeddings de ``params.yaml``, celui qui encode les questions.
"""

import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.config import Params, get_secret  # noqa: E402
from src.pipeline.build_index import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE  # noqa: E402
from src.pipeline.utils import read_json  # noqa: E402

logger = logging.getLogger(__name__)

TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"


@dataclass(frozen=True)
class ServedIndex:
    directory: Path
    source: str                 # "mlflow" ou "local"
    version: str | None         # version du modèle enregistré (source mlflow)
    run_id: str | None
    manifest: dict[str, Any]

    @property
    def index_path(self) -> Path:
        return self.directory / INDEX_FILE

    @property
    def chunks_path(self) -> Path:
        return self.directory / CHUNKS_FILE

    def describe(self) -> dict[str, Any]:
        info = asdict(self)
        info["directory"] = str(self.directory)
        return info


def check_manifest(directory: Path, params: Params) -> dict[str, Any]:
    """Refuse un index incomplet ou encodé avec un autre modèle que celui des questions."""
    missing = [name for name in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE) if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"index incomplet dans {directory} : {', '.join(missing)} absent(s)")
    manifest = read_json(directory / MANIFEST_FILE)
    if manifest.get("embedding_model") != params.embedding.model:
        raise ValueError(
            f"index encodé avec {manifest.get('embedding_model')!r}, "
            f"mais les questions le seraient avec {params.embedding.model!r}"
        )
    return manifest


def fetch_champion(params: Params) -> ServedIndex:
    """Télécharge la version champion dans ``paths.served_index_dir`` (remplace le contenu)."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ[TRACKING_URI_VAR])
    name, alias = params.mlflow.registered_model, params.mlflow.champion_alias
    version = MlflowClient().get_model_version_by_alias(name, alias)

    target = params.paths.served_index_dir
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    # Version explicite (et non l'alias) : l'alias peut bouger pendant le téléchargement.
    downloaded = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version.version}", dst_path=str(staging)
    ))
    manifest = check_manifest(downloaded, params)
    # Remplacement en fin de téléchargement : un échec laisse l'index précédent intact.
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(downloaded), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    return ServedIndex(target, "mlflow", str(version.version), version.run_id, manifest)


def load_served_index(params: Params) -> ServedIndex:
    if get_secret(TRACKING_URI_VAR) is not None:  # charge aussi les identifiants MLflow de .env
        served = fetch_champion(params)
    else:
        directory = params.paths.index_dir
        served = ServedIndex(directory, "local", None, None, check_manifest(directory, params))
    logger.info("index_served", extra=served.describe())
    return served
