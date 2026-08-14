from api.main import app
from fastapi.testclient import TestClient
import sys
from pathlib import Path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_minimal_data():
    """Тест: отправляем ТОЛЬКО 5 обязательных полей"""
    payload = {
        "GrLivArea": 2000,
        "OverallQual": 7,
        "YearBuilt": 2005,
        "LotArea": 8500,
        "OverallCond": 5
    }
    response = client.post("/predict", json=payload)

    print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert data["currency"] == "USD"


def test_predict_validation_error():
    """Тест: не отправляем обязательное поле (GrLivArea) -> ждем 422 ошибку"""
    payload = {
        "OverallQual": 7,
        "YearBuilt": 2005,
        "LotArea": 8500,
        "OverallCond": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Error: Unprocessable Entity
