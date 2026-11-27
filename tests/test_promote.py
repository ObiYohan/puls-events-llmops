import mlflow
import pytest
from mlflow import MlflowClient

from src.config import PROJECT_ROOT, get_params
from src.pipeline import promote
from src.pipeline.evaluate import METRICS
from src.pipeline.utils import read_json, write_json


def evaluation(score=0.95, valid=True, version=2, **overrides):
    result = {name: score for name in METRICS}
    result |= {
        "valid": valid, "n_metric_errors": 0 if valid else 1, "error_rate": 0.0 if valid else 0.01,
        "n_cases": 24, "reference_set_version": version, "evaluated_on": "2026-11-27",
        "judge_model": "gpt-5-mini", "judge_reasoning_effort": "minimal",
        "tokens": {
            "generation": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000, "embedding_tokens": 1_000_000},
            "judge": {"prompt_tokens": 2_000_000, "completion_tokens": 500_000, "reasoning_tokens": 0, "calls": 10},
        },
    }
    return result | overrides


@pytest.fixture
def params():
    return get_params()


def candidate(params, **kwargs):
    ev = evaluation(**kwargs)
    return ev | {"judge_model": params.evaluation.judge_model, "judge_reasoning_effort": params.evaluation.judge_reasoning_effort}


def champion(params, score=0.95, **overrides):
    base = {name: str(score) for name in METRICS}
    return base | {
        "judge_model": params.evaluation.judge_model,
        "judge_reasoning_effort": params.evaluation.judge_reasoning_effort,
        "reference_set_version": "2", "version": "3",
    } | overrides


# --- decide ------------------------------------------------------------------

def test_invalid_evaluation_is_never_promoted(params):
    decision = promote.decide(candidate(params, valid=False), None, params)
    assert not decision.promote and "invalide" in decision.reasons[0]


def test_threshold_miss_is_rejected_using_current_thresholds(params):
    low = params.evaluation.thresholds.contextual_precision - 0.01
    decision = promote.decide(candidate(params, contextual_precision=low), None, params)
    assert not decision.promote and "contextual_precision" in decision.reasons[0]


def test_first_version_is_promoted(params):
    decision = promote.decide(candidate(params), None, params)
    assert decision.promote and not decision.compared_to_champion


def test_small_drop_within_tolerance_is_promoted(params):
    score = 0.95 - params.promotion.tolerance / 2
    decision = promote.decide(candidate(params, score=score), champion(params), params)
    assert decision.promote and decision.compared_to_champion


def test_regression_beyond_tolerance_is_rejected(params):
    worse = 0.95 - params.promotion.tolerance - 0.01
    decision = promote.decide(candidate(params, faithfulness=worse), champion(params), params)
    assert not decision.promote
    assert "faithfulness" in decision.reasons[0] and decision.compared_to_champion


@pytest.mark.parametrize("override", [{"judge_model": "autre-juge"}, {"reference_set_version": "1"}])
def test_champion_with_other_judge_or_reference_set_is_not_compared(params, override):
    worse = 0.95 - params.promotion.tolerance - 0.01
    decision = promote.decide(candidate(params, score=worse), champion(params, **override), params)
    assert decision.promote and not decision.compared_to_champion
    assert "comparaison impossible" in decision.reasons[0]


# --- coût, paramètres, traçabilité --------------------------------------------

def test_cost_uses_pricing_per_model(params):
    cost = promote.cost_usd(evaluation()["tokens"], params)
    gen, emb, judge = (params.pricing[m] for m in (params.generation.model, params.embedding.model, params.evaluation.judge_model))
    assert cost["cost_usd_generation"] == pytest.approx(gen.input + 0.1 * gen.output + emb.input)
    assert cost["cost_usd_judge"] == pytest.approx(2 * judge.input + 0.5 * judge.output)
    assert cost["cost_usd_total"] == pytest.approx(cost["cost_usd_generation"] + cost["cost_usd_judge"])


def test_missing_price_is_rejected_by_config(params):
    from src.config import Params

    data = params.model_dump()
    del data["pricing"][params.evaluation.judge_model]
    with pytest.raises(ValueError, match="prix manquant"):
        Params.model_validate(data)


def test_run_params_include_prompt_hash_and_comparability_keys(params):
    logged = promote.run_params(params, evaluation())
    assert len(logged["prompt_hash"]) == 16
    assert all(key in logged for key in promote.COMPARABILITY_KEYS)


def test_index_md5_reads_dvc_lock(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text("stages:\n  build_index:\n    outs:\n    - path: data/processed/index\n      md5: abc.dir\n", encoding="utf-8")
    assert promote.index_md5(lock) == "abc.dir"
    assert promote.index_md5(tmp_path / "absent.lock") == "inconnu"


# --- étape complète sur un MLflow local (SQLite, sans réseau) -----------------

@pytest.fixture
def local_mlflow(isolated_params, tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv(promote.TRACKING_URI_VAR, uri)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(isolated_params.mlflow.experiment, artifact_location=(tmp_path / "artifacts").as_uri())
    params = isolated_params.model_copy(update={
        "paths": isolated_params.paths.model_copy(update={"system_prompt": PROJECT_ROOT / "src/prompts/system_prompt.txt"}),
    })
    monkeypatch.setattr(promote, "get_params", lambda: params)
    write_json(params.paths.eval_cases, [{"id": "a"}])
    write_json(params.paths.index_dir / "manifest.json", {"dim": 3})
    yield params
    mlflow.set_tracking_uri("")


def write_evaluation(params, **kwargs):
    ev = evaluation(**kwargs) | {
        "judge_model": params.evaluation.judge_model, "judge_reasoning_effort": params.evaluation.judge_reasoning_effort,
    }
    write_json(params.paths.eval_metrics, ev)


def test_run_promotes_first_version_and_sets_alias(local_mlflow):
    params = local_mlflow
    write_evaluation(params)

    report = promote.run()

    assert report["promote"] and report["new_version"] == "1" and report["previous_champion"] is None
    client = MlflowClient()
    version = client.get_model_version_by_alias(params.mlflow.registered_model, params.mlflow.champion_alias)
    assert str(version.version) == "1"
    run = client.get_run(report["run_id"])
    assert run.data.tags["decision"] == "promoted"
    assert run.data.metrics["faithfulness"] == 0.95
    assert run.data.metrics["cost_usd_total"] > 0
    assert run.data.params["judge_model"] == params.evaluation.judge_model
    files = [f.path for f in client.list_artifacts(report["run_id"], "index")]
    assert files == ["index/manifest.json"]
    assert read_json(params.paths.promotion_report) == report


def test_run_rejects_regression_and_keeps_champion(local_mlflow):
    params = local_mlflow
    write_evaluation(params, score=0.95)
    promote.run()
    write_evaluation(params, score=0.95, faithfulness=0.95 - params.promotion.tolerance - 0.02)

    report = promote.run()

    assert not report["promote"] and report["new_version"] is None
    assert report["previous_champion"] == "1"
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(params.mlflow.registered_model, params.mlflow.champion_alias).version) == "1"
    assert client.get_run(report["run_id"]).data.tags["decision"] == "rejected"
    assert client.list_artifacts(report["run_id"], "index") == []  # index non téléversé si refus


def test_run_moves_alias_to_better_version(local_mlflow):
    params = local_mlflow
    write_evaluation(params, score=0.90)
    promote.run()
    write_evaluation(params, score=0.97)

    report = promote.run()

    assert report["promote"] and report["new_version"] == "2" and report["previous_champion"] == "1"


def test_run_requires_tracking_uri(local_mlflow, monkeypatch):
    monkeypatch.delenv(promote.TRACKING_URI_VAR)
    with pytest.raises(RuntimeError, match=promote.TRACKING_URI_VAR):
        promote.run()
