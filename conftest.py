import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--csv", action="store", default=None, help="Chemin vers le fichier CSV à tester"
    )

@pytest.fixture
def csv_file(request):
    csv_path = request.config.getoption("--csv")
    if not csv_path:
        pytest.fail("Veuillez fournir un fichier CSV avec l'option --csv")
    return csv_path
