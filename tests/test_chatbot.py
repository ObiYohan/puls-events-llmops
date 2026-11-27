import json

import faiss
import numpy as np
import pandas as pd
import pytest

from src.chatbot import EventChatbot, parse_dates

DIMENSION = 4


def write_artifacts(tmp_path, chunks_data, n_vectors=None):
    """Écrit un index FAISS (un vecteur par chunk) et le JSON des chunks."""
    index_path = tmp_path / "test_index.idx"
    chunks_path = tmp_path / "test_chunks.json"
    n = len(chunks_data) if n_vectors is None else n_vectors
    index = faiss.IndexFlatL2(DIMENSION)
    if n:
        index.add(np.full((n, DIMENSION), 0.1, dtype=np.float32))
    faiss.write_index(index, str(index_path))
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    return index_path, chunks_path


def test_init_with_valid_paths(tmp_path):
    """Test successful initialization with valid index and chunks paths"""
    # Create sample chunks data with required columns
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event description",
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "firstdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    index_path, chunks_path = write_artifacts(tmp_path, chunks_data)
    
    # Initialize chatbot with explicit paths
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Assertions
    assert chatbot.index is not None
    assert chatbot.index.ntotal == 1
    assert chatbot.embeddings.shape == (1, DIMENSION)
    assert isinstance(chatbot.chunks_df, pd.DataFrame)
    assert len(chatbot.chunks_df) == 1
    assert 'uid' in chatbot.chunks_df.columns
    assert 'text' in chatbot.chunks_df.columns
    assert chatbot.available_cities is not None
    assert chatbot.available_postal_codes is not None
    assert chatbot.client is not None

def test_init_raises_error_when_index_path_does_not_exist(tmp_path):
    """Test that FileNotFoundError is raised when index_path does not exist"""
    # Create a non-existent index path
    non_existent_index = tmp_path / "non_existent_index.idx"
    
    # Create a valid chunks file
    chunks_path = tmp_path / "test_chunks.json"
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event",
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
        }
    ]
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f)
    
    # Attempt to initialize with non-existent index path
    try:
        chatbot = EventChatbot(index_path=str(non_existent_index), chunks_path=str(chunks_path))
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError as e:
        assert "Index FAISS introuvable" in str(e)
        assert str(non_existent_index) in str(e)

def test_init_raises_error_when_chunks_path_does_not_exist(tmp_path):
    """Test that FileNotFoundError is raised when chunks_path does not exist"""
    # Create a valid FAISS index
    index_path = tmp_path / "test_index.idx"
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Create a non-existent chunks path
    non_existent_chunks = tmp_path / "non_existent_chunks.json"
    
    # Attempt to initialize with non-existent chunks path
    try:
        chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(non_existent_chunks))
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError as e:
        assert "Fichier chunks introuvable" in str(e)
        assert str(non_existent_chunks) in str(e)


def test_format_context_should_format_multiday_event_with_warning(tmp_path):
    """Test that multi-day events are formatted with explicit warnings about start and end dates"""
    # Create a multi-day event (different dates for firstdate_begin and lastdate_end)
    now = pd.Timestamp.now(tz='UTC')
    start_date = now + pd.Timedelta(days=5)  # Starts in 5 days
    end_date = now + pd.Timedelta(days=8)    # Ends 3 days later (multi-day event)
    
    chunks_data = [
        {
            "uid": "multiday_event_1",
            "chunk_id": 0,
            "text": "Festival de musique sur plusieurs jours",
            "firstdate_begin": start_date.timestamp() * 1000,
            "firstdate_end": (start_date + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": (end_date - pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_end": end_date.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_name": "Parc des expositions"
        }
    ]
    
    index_path, chunks_path = write_artifacts(tmp_path, chunks_data)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Format context for the multi-day event
    formatted_context = chatbot.format_context(chatbot.chunks_df)
    
    # Assertions
    assert "⚠️ ÉVÉNEMENT MULTI-JOURS" in formatted_context
    assert "Jour de début:" in formatted_context
    assert "Jour de fin:" in formatted_context
    assert "⚠️ NE PAS CONFONDRE la date de début et la date de fin" in formatted_context
    assert start_date.tz_convert('Europe/Paris').strftime('%d/%m/%Y') in formatted_context
    assert end_date.tz_convert('Europe/Paris').strftime('%d/%m/%Y') in formatted_context
    assert "=== ÉVÉNEMENT multiday_event_1 ===" in formatted_context
    assert "DATES:" in formatted_context

def test_format_context_should_format_single_day_event(tmp_path):
    """Test that single-day events are formatted without multi-day warnings"""
    # Create a single-day event (same date for firstdate_begin and firstdate_end)
    now = pd.Timestamp.now(tz='UTC')
    event_date = now + pd.Timedelta(days=5)  # Starts in 5 days
    event_end = event_date + pd.Timedelta(hours=2)  # Ends same day, 2 hours later
    
    chunks_data = [
        {
            "uid": "single_day_event_1",
            "chunk_id": 0,
            "text": "Concert de jazz en soirée",
            "firstdate_begin": event_date.timestamp() * 1000,
            "firstdate_end": event_end.timestamp() * 1000,
            "lastdate_begin": event_date.timestamp() * 1000,
            "lastdate_end": event_end.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001",
            "location_name": "Salle Pleyel"
        }
    ]
    
    index_path, chunks_path = write_artifacts(tmp_path, chunks_data)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Format context for the single-day event
    formatted_context = chatbot.format_context(chatbot.chunks_df)
    
    # Assertions
    assert "⚠️ ÉVÉNEMENT MULTI-JOURS" not in formatted_context
    assert "Jour de début:" not in formatted_context
    assert "Jour de fin:" not in formatted_context
    assert "⚠️ NE PAS CONFONDRE la date de début et la date de fin" not in formatted_context
    assert "Date début:" in formatted_context
    assert "Date fin:" in formatted_context
    assert event_date.tz_convert('Europe/Paris').strftime('%d/%m/%Y') in formatted_context
    assert "=== ÉVÉNEMENT single_day_event_1 ===" in formatted_context
    assert "DATES:" in formatted_context


def test_interactive_mode_should_print_welcome_message_with_banner_when_started(tmp_path, capsys, monkeypatch):
    """Test that interactive mode prints welcome message with banner when started"""
    # Create sample chunks data
    chunks_data = [
        {
            "uid": "event1",
            "chunk_id": 0,
            "text": "Sample event description",
            "firstdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "firstdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": pd.Timestamp.now(tz='UTC').timestamp() * 1000,
            "lastdate_end": (pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=30)).timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    index_path, chunks_path = write_artifacts(tmp_path, chunks_data)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Mock user input to immediately quit
    monkeypatch.setattr('builtins.input', lambda _: 'quit')
    
    # Clear any previous output
    capsys.readouterr()
    
    # Run interactive mode
    chatbot.interactive_mode()
    
    # Capture output
    captured = capsys.readouterr()
    
    # Assertions
    assert "="*80 in captured.out
    assert "🎭 CHATBOT D'ÉVÉNEMENTS CULTURELS" in captured.out
    assert "Tapez 'quit' ou 'exit' pour quitter" in captured.out
    assert "👋 Au revoir !" in captured.out


def test_apply_temporal_filter_should_correctly_filter_events_that_start_before_target_month_and_end_during_target_month(tmp_path):
    """Test that events starting before the target month and ending during it are correctly filtered"""
    # Create test data with an event that starts before March and ends during March
    now = pd.Timestamp.now(tz='UTC')
    
    # Event that starts in February and ends in March
    event_start = pd.Timestamp(year=2025, month=2, day=20, tz='UTC')
    event_end = pd.Timestamp(year=2025, month=3, day=10, tz='UTC')
    
    chunks_data = [
        {
            "uid": "event_spanning_months",
            "chunk_id": 0,
            "text": "Event spanning from February to March",
            "firstdate_begin": event_start.timestamp() * 1000,
            "firstdate_end": (event_start + pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_begin": (event_end - pd.Timedelta(hours=2)).timestamp() * 1000,
            "lastdate_end": event_end.timestamp() * 1000,
            "location_city": "Paris",
            "location_postalcode": "75001"
        }
    ]
    
    index_path, chunks_path = write_artifacts(tmp_path, chunks_data)
    
    # Initialize chatbot
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    
    # Create temporal filter for March 2025
    temporal_filter = {'type': 'month', 'month': 3, 'year': 2025}
    
    # Apply temporal filter
    filtered_df = chatbot._apply_temporal_filter(chatbot.chunks_df, temporal_filter)
    
    # Assertions
    assert len(filtered_df) == 1, "Event starting before March and ending in March should be included"
    assert filtered_df.iloc[0]['uid'] == "event_spanning_months"
    assert filtered_df.iloc[0]['firstdate_begin'] < pd.Timestamp(year=2025, month=3, day=1, tz='UTC')
    assert filtered_df.iloc[0]['lastdate_end'] >= pd.Timestamp(year=2025, month=3, day=1, tz='UTC')

# --- Format produit par l'étape build_index ------------------------------------

def iso_event(uid="e1", city="Digne-les-Bains", postal="04000", begin="2099-12-05T20:00:00+01:00"):
    return {
        "uid": uid,
        "chunk_id": 0,
        "text": "Concert",
        "firstdate_begin": begin,
        "firstdate_end": "2099-12-05T23:00:00+01:00",
        "lastdate_begin": begin,
        "lastdate_end": "2099-12-05T23:00:00+01:00",
        "location_city": city,
        "location_postalcode": postal,
    }


def test_parse_dates_accepts_iso_and_epoch_ms():
    iso = parse_dates(pd.Series(["2026-12-05T20:00:00+01:00"]))
    ms = parse_dates(pd.Series([pd.Timestamp("2026-12-05T19:00:00Z").timestamp() * 1000]))
    assert iso.iloc[0] == ms.iloc[0] == pd.Timestamp("2026-12-05T19:00:00Z")


def test_iso_dates_are_parsed_and_future_events_kept(tmp_path):
    index_path, chunks_path = write_artifacts(tmp_path, [iso_event()])
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    assert chatbot.chunks_df["lastdate_end"].iloc[0] == pd.Timestamp("2099-12-05T22:00:00Z")


def test_postal_code_keeps_leading_zero(tmp_path):
    index_path, chunks_path = write_artifacts(tmp_path, [iso_event(postal="04000")])
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    assert chatbot.chunks_df["location_postalcode"].iloc[0] == "04000"
    assert "04000" in chatbot.available_postal_codes


def test_format_context_shows_paris_local_time(tmp_path):
    index_path, chunks_path = write_artifacts(tmp_path, [iso_event(begin="2099-12-05T20:00:00+01:00")])
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    context = chatbot.format_context(chatbot.chunks_df)
    assert "05/12/2099 20:00" in context  # et non 19:00 (UTC)


def test_misaligned_index_and_chunks_are_rejected(tmp_path):
    index_path, chunks_path = write_artifacts(tmp_path, [iso_event("a"), iso_event("b")], n_vectors=1)
    with pytest.raises(ValueError, match="désalignés"):
        EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))


def test_semantic_rank_uses_vectors_from_faiss_index(tmp_path):
    chunks = [iso_event("near"), iso_event("far")]
    index_path = tmp_path / "idx"
    chunks_path = tmp_path / "chunks.json"
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))

    result, distances = chatbot._semantic_rank(chatbot.chunks_df, np.array([[0.9, 0.1]], dtype=np.float32), k=1)

    assert list(result["uid"]) == ["near"]
    assert len(distances) == 1


# --- extract_temporal_filter ---------------------------------------------------

@pytest.fixture
def bot(tmp_path):
    index_path, chunks_path = write_artifacts(tmp_path, [iso_event()])
    return EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))


def test_month_inside_a_word_is_not_a_month(bot):
    # « mars » est contenu dans « Marseille ».
    assert bot.extract_temporal_filter("Un concert à Marseille") is None
    assert bot.extract_temporal_filter("Un concert à Marseille en décembre")["month"] == 12


def test_past_month_without_year_means_next_year(bot, monkeypatch):
    import src.chatbot as chatbot_module

    class FrozenDatetime(chatbot_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 27)

    monkeypatch.setattr(chatbot_module, "datetime", FrozenDatetime)
    assert bot.extract_temporal_filter("expo en mars") == {"type": "month", "month": 3, "year": 2027}
    assert bot.extract_temporal_filter("expo en décembre") == {"type": "month", "month": 12, "year": 2026}
    assert bot.extract_temporal_filter("expo en mars 2026")["year"] == 2026


# --- extract_location_filter -----------------------------------------------------

@pytest.fixture
def bot_cities(tmp_path):
    events = [iso_event(f"e{i}", city=city) for i, city in enumerate(["Vence", "Aix-en-Provence", "Aix", "Nice"])]
    index_path, chunks_path = write_artifacts(tmp_path, events)
    return EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))


@pytest.mark.parametrize(("query", "expected"), [
    ("Des spectacles variés ce mois-ci", None),               # « var » dans « variés »
    ("Un festival en Provence", None),                         # « vence » dans « provence »
    ("Un concert à Vence", {"type": "city", "value": "Vence"}),
    ("Expositions à Aix-en-Provence", {"type": "city", "value": "Aix-en-Provence"}),
    ("Que faire dans le Var ?", {"type": "department_prefix", "value": "83"}),
    ("Sortir à Nice", {"type": "city", "value": "Nice"}),
])
def test_location_filter_matches_whole_words_only(bot_cities, query, expected):
    assert bot_cities.extract_location_filter(query) == expected


# --- chat : aucune trace de la question (B7) -------------------------------------

def test_chat_neither_prints_nor_logs_the_question(bot, capsys, caplog):
    from types import SimpleNamespace

    secret = "question confidentielle de Jeanne Dupont"
    bot.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * DIMENSION)], usage=SimpleNamespace(prompt_tokens=7))),
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Voici un concert."))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5)))),
    )
    caplog.set_level("DEBUG")

    result = bot.chat(secret)

    assert result["response"] == "Voici un concert."
    assert capsys.readouterr().out == ""
    assert "Jeanne" not in caplog.text
    assert bot.token_usage == {"embedding_tokens": 7, "prompt_tokens": 50, "completion_tokens": 5}


def test_format_context_shows_department(tmp_path):
    event = {**iso_event(), "location_department": "Vaucluse"}
    index_path, chunks_path = write_artifacts(tmp_path, [event])
    chatbot = EventChatbot(index_path=str(index_path), chunks_path=str(chunks_path))
    assert "Département: Vaucluse" in chatbot.format_context(chatbot.chunks_df)
