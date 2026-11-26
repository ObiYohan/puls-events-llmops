import pytest

from src.pipeline import chunk
from src.pipeline.utils import count_tokens, read_json, text_hash, write_json


@pytest.fixture
def chunking(isolated_params):
    return isolated_params.chunking


def event(uid="1", description="Concert de jazz au bord de l'eau.", **extra):
    return {
        "uid": uid,
        "longdescription_fr": description,
        "firstdate_begin": "2026-12-05T20:00:00+01:00",
        "firstdate_end": "2026-12-05T23:00:00+01:00",
        "lastdate_begin": "2026-12-05T20:00:00+01:00",
        "lastdate_end": "2026-12-05T23:00:00+01:00",
        "location_name": "Théâtre Silvain",
        "location_city": "Marseille",
        "location_postalcode": "13007",
        "title_fr": "Jazz",  # champ hors métadonnées de chunk
        **extra,
    }


def long_description(n_sentences=400):
    return " ".join(f"Phrase numéro {i} décrivant le programme du festival." for i in range(n_sentences))


# --- enriched_text -----------------------------------------------------------

def test_enriched_text_adds_city_place_and_french_month():
    text = chunk.enriched_text("Concert", event())
    assert text == "Concert | Ville: Marseille | Lieu: Théâtre Silvain | Date: décembre 2026"


def test_enriched_text_uses_event_timezone_for_month():
    # 31/12 23:30 à -01:00 = 01/01 en UTC : le mois affiché reste décembre.
    text = chunk.enriched_text("Réveillon", {"firstdate_begin": "2026-12-31T23:30:00-01:00"})
    assert text == "Réveillon | Date: décembre 2026"


@pytest.mark.parametrize("value", [None, "", "  "])
def test_enriched_text_skips_missing_metadata(value):
    metadata = {"location_city": value, "location_name": value, "firstdate_begin": value}
    assert chunk.enriched_text("Concert", metadata) == "Concert"


def test_enriched_text_ignores_invalid_date():
    assert chunk.enriched_text("Concert", {"firstdate_begin": "pas une date"}) == "Concert"


# --- chunk_event -------------------------------------------------------------

def test_short_description_gives_single_chunk_with_metadata(chunking):
    chunks = chunk.chunk_event(event(), chunking, chunk.make_splitter(chunking))

    assert len(chunks) == 1
    c = chunks[0]
    assert c["uid"] == "1" and c["chunk_id"] == 0
    assert c["text"] == event()["longdescription_fr"]
    assert c["location_city"] == "Marseille"
    assert c["lastdate_end"] == "2026-12-05T23:00:00+01:00"
    assert "title_fr" not in c
    assert c["text_hash"] == text_hash(c["text_for_embedding"])


def test_long_description_is_split_and_keeps_metadata(chunking):
    description = long_description()
    assert count_tokens(description, chunking.tokenizer) > chunking.split_threshold

    chunks = chunk.chunk_event(event(description=description), chunking, chunk.make_splitter(chunking))

    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c["token_count"] <= chunking.chunk_size for c in chunks)
    assert all(c["location_city"] == "Marseille" for c in chunks)
    assert len({c["text_hash"] for c in chunks}) == len(chunks)


def test_description_at_threshold_is_not_split(chunking):
    params = chunking.model_copy(update={"split_threshold": 10_000})
    chunks = chunk.chunk_event(event(description=long_description()), params, chunk.make_splitter(params))
    assert len(chunks) == 1


def test_hash_changes_with_enriched_metadata(chunking):
    splitter = chunk.make_splitter(chunking)
    marseille = chunk.chunk_event(event(), chunking, splitter)[0]["text_hash"]
    nice = chunk.chunk_event(event(location_city="Nice"), chunking, splitter)[0]["text_hash"]
    again = chunk.chunk_event(event(), chunking, splitter)[0]["text_hash"]
    assert marseille != nice
    assert marseille == again


# --- run ---------------------------------------------------------------------

def test_run_writes_sorted_chunks_and_metrics(isolated_params, monkeypatch):
    monkeypatch.setattr(chunk, "get_params", lambda: isolated_params)
    write_json(isolated_params.paths.raw_events, [event("b", description=long_description()), event("a")])

    metrics = chunk.run()

    chunks = read_json(isolated_params.paths.chunks_dataset)
    assert [(c["uid"], c["chunk_id"]) for c in chunks][:2] == [("a", 0), ("b", 0)]
    assert read_json(isolated_params.paths.chunk_metrics) == metrics
    assert metrics["n_events"] == 2
    assert metrics["n_chunks"] == len(chunks)
    assert metrics["n_split_events"] == 1
    assert metrics["chunk_tokens_max"] <= isolated_params.chunking.split_threshold
