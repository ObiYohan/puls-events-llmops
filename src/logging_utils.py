"""Logs structurés au format JSON (une ligne par événement).

Les champs passés via ``extra={...}`` sont ajoutés tels quels à l'objet JSON :
``logger.info("ingest_done", extra={"n_events": 1431})``.
"""

import json
import logging
from datetime import UTC, datetime

# Attributs standard d'un LogRecord, à ne pas recopier dans le JSON.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure le logger racine pour écrire du JSON sur la sortie d'erreur."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # httpx (et httpx2, utilisé par le SDK OpenAI) journalisent chaque requête en
    # INFO : trop bavard pour le pipeline.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
