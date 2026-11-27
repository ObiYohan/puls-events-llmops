import faiss
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from src import serving
from src.pipeline.build_index import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE
from src.pipeline.utils import read_json, write_json


def write_index(directory, model, n=2):
    directory.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatL2(3)
    index.add(np.ones((n, 3), dtype=np.float32))
    faiss.write_index(index, str(directory / INDEX_FILE))
    write_json(directory / CHUNKS_FILE, [{"uid": str(i)} for i in range(n)])
    write_json(directory / MANIFEST_FILE, {"embedding_model": model, "dim": 3, "n_vectors": n})


def test_local_index_is_served_without_mlflow(isolated_params, monkeypatch):
    monkeypatch.delenv(serving.TRACKING_URI_VAR, raising=False)
    write_index(isolated_params.paths.index_dir, isolated_params.embedding.model)

    served = serving.load_served_index(isolated_params)

    assert served.source == "local" and served.version is None
    assert served.index_path == isolated_params.paths.index_dir / INDEX_FILE


def test_index_from_another_embedding_model_is_refused(isolated_params, monkeypatch):
    monkeypatch.delenv(serving.TRACKING_URI_VAR, raising=False)
    write_index(isolated_params.paths.index_dir, "autre-modele")
    with pytest.raises(ValueError, match="autre-modele"):
        serving.load_served_index(isolated_params)


def test_incomplete_index_is_refused(isolated_params, monkeypatch):
    monkeypatch.delenv(serving.TRACKING_URI_VAR, raising=False)
    write_index(isolated_params.paths.index_dir, isolated_params.embedding.model)
    (isolated_params.paths.index_dir / CHUNKS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=CHUNKS_FILE):
        serving.load_served_index(isolated_params)


@pytest.fixture
def registry(isolated_params, tmp_path, monkeypatch):
    """Registre MLflow local (SQLite) avec deux versions ; l'alias champion pointe la v2."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv(serving.TRACKING_URI_VAR, uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("exp", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("exp")
    client = MlflowClient()
    name = isolated_params.mlflow.registered_model
    client.create_registered_model(name)
    for n in (1, 2):
        source = tmp_path / f"index-v{n}"
        write_index(source, isolated_params.embedding.model, n=n)
        with mlflow.start_run() as run:
            mlflow.log_artifacts(str(source), artifact_path="index")
        version = client.create_model_version(name, source=f"{run.info.artifact_uri}/index", run_id=run.info.run_id)
    client.set_registered_model_alias(name, isolated_params.mlflow.champion_alias, version.version)
    yield isolated_params
    mlflow.set_tracking_uri("")


def test_champion_is_downloaded_from_registry(registry):
    served = serving.load_served_index(registry)

    assert served.source == "mlflow" and served.version == "2"
    assert served.directory == registry.paths.served_index_dir
    assert faiss.read_index(str(served.index_path)).ntotal == 2
    assert read_json(served.chunks_path) == [{"uid": "0"}, {"uid": "1"}]


def test_reload_replaces_previous_download(registry):
    first = serving.load_served_index(registry)
    (first.directory / "fichier-obsolete").write_text("x", encoding="utf-8")
    second = serving.load_served_index(registry)
    assert not (second.directory / "fichier-obsolete").exists()
    assert not registry.paths.served_index_dir.with_name("index.tmp").exists()


def test_unreachable_or_empty_registry_fails_loudly(registry):
    MlflowClient().delete_registered_model_alias(registry.mlflow.registered_model, registry.mlflow.champion_alias)
    with pytest.raises(Exception):
        serving.load_served_index(registry)
