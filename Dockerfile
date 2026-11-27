# API du chatbot Puls-Events.
# Aucune donnée dans l'image : au démarrage, l'API télécharge l'index « champion »
# du registre MLflow (src/serving.py). Secrets et configuration par variables
# d'environnement (AZURE_OPENAI_*, MLFLOW_TRACKING_*, ADMIN_API_KEY, ALLOWED_ORIGINS).

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Dépendances du projet seulement : ni DVC, ni DeepEval, ni outils de test.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --no-install-project && rm -rf /root/.cache

COPY params.yaml ./
COPY src/ ./src/
COPY api/ ./api/

# Utilisateur sans privilèges ; seul data/ (index téléchargé) est inscriptible.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data \
    && chown app:app data
USER app

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
