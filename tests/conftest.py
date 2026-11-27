import sys
from pathlib import Path
import pytest

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def isolated_params(tmp_path):
    """Paramètres du projet avec tous les chemins redirigés sous tmp_path."""
    from src.config import PROJECT_ROOT, get_params

    params = get_params()
    paths = params.paths.model_copy(update={
        name: tmp_path / value.relative_to(PROJECT_ROOT) for name, value in params.paths
    })
    return params.model_copy(update={"paths": paths})


@pytest.fixture(autouse=True)
def ignore_local_dotenv(monkeypatch):
    """Empêche le .env local (vraies clés) d'être chargé pendant les tests."""
    monkeypatch.setattr("src.config._load_dotenv", lambda: None)


@pytest.fixture(autouse=True)
def set_test_env_vars(monkeypatch):
    """Secrets factices : aucun appel réseau n'est fait avec ces valeurs."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test_azure_key_12345")