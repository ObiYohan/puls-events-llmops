import json
import logging

from src.logging_utils import JsonFormatter, setup_logging


def make_record(**extra):
    record = logging.LogRecord("src.test", logging.INFO, __file__, 1, "ingest_done", None, None)
    record.__dict__.update(extra)
    return record


def test_formatter_outputs_one_json_object_with_extra_fields():
    line = JsonFormatter().format(make_record(n_events=12, region="Provence-Alpes-Côte d'Azur"))
    payload = json.loads(line)
    assert payload["message"] == "ingest_done"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "src.test"
    assert payload["n_events"] == 12
    assert payload["region"] == "Provence-Alpes-Côte d'Azur"
    assert "\n" not in line


def test_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord("src.test", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_setup_logging_installs_json_handler():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        setup_logging()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
