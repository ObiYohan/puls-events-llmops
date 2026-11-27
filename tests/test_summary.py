from src.config import get_params
from src.pipeline import summary
from src.pipeline.evaluate import METRICS
from src.pipeline.utils import write_json

INDEX = {"n_vectors": 1505, "n_events": 1346, "embedding_model": "text-embedding-3-small", "dim": 1536}


def evaluation(score=0.95):
    return {name: score for name in METRICS} | {
        "evaluated_on": "2026-11-27", "judge_model": "gpt-5-mini", "judge_reasoning_effort": "low",
        "n_cases": 24, "valid": True, "n_metric_errors": 0,
    }


def test_render_promoted_run():
    promotion = {"promote": True, "new_version": "2", "registered_model": "puls-events-rag",
                 "previous_champion": "1", "reasons": ["au moins aussi bon que le champion"]}
    text = summary.render(INDEX, evaluation(), promotion, get_params().evaluation.thresholds)
    assert "1505 chunks" in text
    assert "| faithfulness | 0.95 ✅ |" in text
    assert "promue" in text and "version 2" in text


def test_render_rejected_run_and_failed_threshold():
    promotion = {"promote": False, "new_version": None, "registered_model": "puls-events-rag",
                 "previous_champion": "1", "reasons": ["seuil(s) non atteint(s)"]}
    text = summary.render(None, evaluation(score=0.1), promotion, get_params().evaluation.thresholds)
    assert "❌" in text and "refusée" in text and "version 1" in text


def test_run_appends_to_github_summary_and_warns_on_rejection(isolated_params, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(summary, "get_params", lambda: isolated_params)
    write_json(isolated_params.paths.index_metrics, INDEX)
    write_json(isolated_params.paths.eval_metrics, evaluation())
    write_json(isolated_params.paths.promotion_report, {
        "promote": False, "new_version": None, "registered_model": "m", "previous_champion": "1", "reasons": ["recul"],
    })
    target = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))

    summary.run()

    assert "Pipeline RAG" in target.read_text(encoding="utf-8")
    assert "::warning::Version non promue : recul" in capsys.readouterr().out


def test_run_without_outputs_prints_header(isolated_params, monkeypatch, capsys):
    monkeypatch.setattr(summary, "get_params", lambda: isolated_params)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    summary.run()
    assert "Pipeline RAG" in capsys.readouterr().out
