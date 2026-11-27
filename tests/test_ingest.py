import json
from datetime import date

import httpx
import pytest

from src.config import get_params
from src.pipeline import ingest

SNAPSHOT = date(2026, 11, 26)


@pytest.fixture
def ingest_params():
    return get_params().ingest.model_copy(update={"max_retries": 2, "retry_base_seconds": 1.0})


@pytest.fixture
def no_sleep(monkeypatch):
    delays = []
    monkeypatch.setattr(ingest.time, "sleep", delays.append)
    return delays


def make_client(*responses):
    """Client httpx qui renvoie les réponses données, dans l'ordre."""
    calls = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def event(uid, description="Concert de jazz", **extra):
    return {
        "uid": uid, "longdescription_fr": description, "title_fr": f"Événement {uid}",
        "firstdate_begin": "2026-12-01T18:00:00+00:00", "lastdate_end": "2026-12-01T20:00:00+00:00", **extra,
    }


# --- build_query -------------------------------------------------------------

def test_build_query_filters_region_and_end_date(ingest_params):
    query = ingest.build_query(ingest_params, SNAPSHOT)
    assert "location_region=\"Provence-Alpes-Côte d'Azur\"" in query["where"]
    assert "lastdate_end >= date'2026-11-26'" in query["where"]
    assert query["select"].split(",") == ingest_params.fields


# --- fetch_events ------------------------------------------------------------

def test_fetch_events_calls_export_endpoint(ingest_params, no_sleep):
    client, calls = make_client(httpx.Response(200, json=[event("1")]))
    assert ingest.fetch_events(ingest_params, SNAPSHOT, client) == [event("1")]
    assert calls[0].url.path.endswith("/evenements-publics-openagenda/exports/json")
    assert "Provence-Alpes-C%C3%B4te" in str(calls[0].url)  # région encodée en UTF-8
    assert no_sleep == []


def test_fetch_events_retries_with_exponential_backoff(ingest_params, no_sleep):
    client, calls = make_client(
        httpx.Response(429),
        httpx.ConnectTimeout("timeout"),
        httpx.Response(200, json=[event("1")]),
    )
    assert ingest.fetch_events(ingest_params, SNAPSHOT, client) == [event("1")]
    assert len(calls) == 3
    assert no_sleep == [1.0, 2.0]


def test_fetch_events_gives_up_after_max_retries(ingest_params, no_sleep):
    client, calls = make_client(*[httpx.Response(503)] * 3)
    with pytest.raises(RuntimeError, match="3 tentatives"):
        ingest.fetch_events(ingest_params, SNAPSHOT, client)
    assert len(calls) == 3


def test_fetch_events_does_not_retry_client_errors(ingest_params, no_sleep):
    client, calls = make_client(httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        ingest.fetch_events(ingest_params, SNAPSHOT, client)
    assert len(calls) == 1


def test_fetch_events_rejects_non_list_payload(ingest_params, no_sleep):
    client, _ = make_client(httpx.Response(200, json={"error": "oops"}))
    with pytest.raises(ValueError, match="liste JSON"):
        ingest.fetch_events(ingest_params, SNAPSHOT, client)


# --- clean_events ------------------------------------------------------------

def test_clean_events_drops_fields_outside_whitelist(ingest_params):
    raw = [event("1", contributor_email="jane@example.org", contributor_contactname="Jane")]
    events, _ = ingest.clean_events(raw, ingest_params, SNAPSHOT)
    assert not any(key.startswith("contributor_") for key in events[0])
    assert events[0]["uid"] == "1"


@pytest.mark.parametrize("description", [None, "", "   ", 42])
def test_clean_events_drops_events_without_description(ingest_params, description):
    events, stats = ingest.clean_events([event("1", description=description)], ingest_params, SNAPSHOT)
    assert events == []
    assert stats["n_without_description"] == 1


def test_clean_events_keeps_latest_duplicate_and_sorts_by_uid(ingest_params):
    raw = [
        event("2"),
        event("1", description="ancienne", updatedat="2026-01-01T00:00:00+00:00"),
        event("1", description="récente", updatedat="2026-06-01T00:00:00+00:00"),
        event("1", description="plus ancienne", updatedat="2025-01-01T00:00:00+00:00"),
    ]
    events, stats = ingest.clean_events(raw, ingest_params, SNAPSHOT)
    assert [e["uid"] for e in events] == ["1", "2"]
    assert events[0]["longdescription_fr"] == "récente"
    assert stats["n_duplicates"] == 2 and stats["n_kept"] == 2 and stats["n_received"] == 4


@pytest.mark.parametrize(("dates", "counter"), [
    ({"lastdate_end": "2026-11-25T23:00:00+00:00"}, "n_past"),                 # terminé la veille
    ({"firstdate_begin": "2032-01-01T01:00:00+00:00", "lastdate_end": "2032-01-01T03:00:00+00:00"}, "n_beyond_horizon"),
    ({"firstdate_begin": None}, "n_invalid_dates"),
    ({"lastdate_end": "pas une date"}, "n_invalid_dates"),
])
def test_clean_events_drops_past_far_and_undated_events(ingest_params, dates, counter):
    events, stats = ingest.clean_events([event("1", **dates), event("2")], ingest_params, SNAPSHOT)
    assert [e["uid"] for e in events] == ["2"]
    assert stats[counter] == 1


def test_clean_events_keeps_event_ending_on_snapshot_day(ingest_params):
    ending_today = event("1", firstdate_begin="2026-11-26T08:00:00+00:00", lastdate_end="2026-11-26T09:00:00+00:00")
    events, _ = ingest.clean_events([ending_today], ingest_params, SNAPSHOT)
    assert len(events) == 1


def test_clean_events_merges_same_event_published_under_two_uids(ingest_params):
    same = {"title_fr": "Atelier « France Travail » !", "location_name": "Agence Manosque", "location_city": "Manosque"}
    raw = [
        event("33", updatedat="2026-09-25T00:00:00+00:00", **same),
        event("43", updatedat="2026-10-23T00:00:00+00:00", **{**same, "title_fr": "atelier france travail"}),
    ]
    events, stats = ingest.clean_events(raw, ingest_params, SNAPSHOT)
    assert [e["uid"] for e in events] == ["43"]  # la version la plus récente
    assert stats["n_content_duplicates"] == 1


def test_clean_events_tie_on_update_date_keeps_smallest_uid_whatever_the_order(ingest_params):
    same = {"title_fr": "Lecture", "location_name": "Bibliothèque", "updatedat": "2026-10-01T00:00:00+00:00"}
    a, b = event("10", **same), event("1", **same)
    assert ingest.clean_events([a, b], ingest_params, SNAPSHOT)[0][0]["uid"] == "1"
    assert ingest.clean_events([b, a], ingest_params, SNAPSHOT)[0][0]["uid"] == "1"


def test_clean_events_keeps_recurring_series_and_same_event_in_two_places(ingest_params):
    raw = [
        event("1", title_fr="Atelier", location_name="Alcazar"),
        event("2", title_fr="Atelier", location_name="Merlan"),                       # autre lieu
        event("3", title_fr="Atelier", location_name="Alcazar",
              firstdate_begin="2026-12-08T18:00:00+00:00", lastdate_end="2026-12-08T20:00:00+00:00"),  # autre date
        event("4", title_fr=None, location_name="Alcazar"),                            # sans titre : jamais fusionné
        event("5", title_fr=None, location_name="Alcazar"),
    ]
    events, stats = ingest.clean_events(raw, ingest_params, SNAPSHOT)
    assert len(events) == 5 and stats["n_content_duplicates"] == 0


# --- run ---------------------------------------------------------------------

@pytest.fixture
def tmp_params(tmp_path, monkeypatch):
    params = get_params()
    paths = params.paths.model_copy(update={
        "raw_events": tmp_path / "raw" / "events.json",
        "ingest_metrics": tmp_path / "metrics" / "ingest.json",
    })
    tmp = params.model_copy(update={"paths": paths})
    monkeypatch.setattr(ingest, "get_params", lambda: tmp)
    return tmp


def test_run_writes_sorted_events_and_metrics(tmp_params, monkeypatch):
    monkeypatch.setattr(ingest, "fetch_events", lambda *_: [event("b"), event("a"), event("c", description="")])

    metrics = ingest.run(SNAPSHOT)

    events = json.loads(tmp_params.paths.raw_events.read_text(encoding="utf-8"))
    assert [e["uid"] for e in events] == ["a", "b"]
    assert json.loads(tmp_params.paths.ingest_metrics.read_text(encoding="utf-8")) == metrics
    assert metrics["snapshot_date"] == "2026-11-26"
    assert metrics["n_kept"] == 2


def test_run_output_is_deterministic(tmp_params, monkeypatch):
    monkeypatch.setattr(ingest, "fetch_events", lambda *_: [event("b"), event("a")])
    ingest.run(SNAPSHOT)
    first = tmp_params.paths.raw_events.read_bytes()

    monkeypatch.setattr(ingest, "fetch_events", lambda *_: [event("a"), event("b")])
    ingest.run(SNAPSHOT)
    assert tmp_params.paths.raw_events.read_bytes() == first


def test_run_fails_when_no_usable_event(tmp_params, monkeypatch):
    monkeypatch.setattr(ingest, "fetch_events", lambda *_: [event("a", description="")])
    with pytest.raises(RuntimeError, match="aucun événement"):
        ingest.run(SNAPSHOT)
    assert not tmp_params.paths.raw_events.exists()
