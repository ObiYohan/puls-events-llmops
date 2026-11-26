from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import PARAMS_FILE, PROJECT_ROOT, get_secret, load_params


@pytest.fixture
def raw_params():
    with PARAMS_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_params(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_project_params_file_is_valid():
    params = load_params()
    assert params.retrieval.k >= 1
    assert params.chunking.chunk_overlap < params.chunking.chunk_size


def test_relative_paths_are_resolved_from_project_root():
    params = load_params()
    assert params.paths.chunks.is_absolute()
    assert params.paths.chunks.is_relative_to(PROJECT_ROOT)


def test_unknown_key_is_rejected(tmp_path, raw_params):
    raw_params["retrieval"]["typo_k"] = 3
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, raw_params))


def test_missing_section_is_rejected(tmp_path, raw_params):
    del raw_params["generation"]
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, raw_params))


def test_threshold_out_of_range_is_rejected(tmp_path, raw_params):
    raw_params["evaluation"]["thresholds"]["faithfulness"] = 1.5
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, raw_params))


def test_overlap_must_be_smaller_than_chunk_size(tmp_path, raw_params):
    raw_params["chunking"]["chunk_overlap"] = raw_params["chunking"]["chunk_size"]
    with pytest.raises(ValidationError, match="chunk_overlap"):
        load_params(write_params(tmp_path, raw_params))


def test_params_are_immutable():
    params = load_params()
    with pytest.raises(ValidationError):
        params.retrieval.k = 99


def test_get_secret_returns_value(monkeypatch):
    monkeypatch.setenv("SOME_SECRET", "  s3cr3t  ")
    assert get_secret("SOME_SECRET") == "s3cr3t"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_get_secret_has_no_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SOME_SECRET", raising=False)
    else:
        monkeypatch.setenv("SOME_SECRET", value)
    assert get_secret("SOME_SECRET") is None
