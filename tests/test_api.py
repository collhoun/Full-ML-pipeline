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


def test_web_ui_is_served():
    """Тест: корневой маршрут отдаёт HTML-страницу интерфейса"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Оценка стоимости дома" in response.text


def test_static_assets_are_served():
    """Тест: css и js интерфейса доступны"""
    for path in ("/static/style.css", "/static/app.js"):
        response = client.get(path)
        assert response.status_code == 200, path


def test_ui_units_payload_matches_api_contract():
    """Тест: конвертация м² -> кв. футы из интерфейса даёт валидный запрос к API"""
    sqm = 159
    payload = {
        "GrLivArea": round(sqm * 10.7639, 2),
        "OverallQual": 7,
        "YearBuilt": 2003,
        "LotArea": 8450.0,
        "OverallCond": 5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0
