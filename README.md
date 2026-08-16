# Ames Housing — ML-пайплайн для оценки стоимости недвижимости

Полный цикл ML-проекта: от исследовательского анализа данных (EDA) до production-бэкенда на **FastAPI**, упакованного в **Docker**. Проект построен по методологии **MLOps** с разделением research-ноутбуков и production-кода, использованием GitFlow, модульных тестов (TDD) и борьбой с типовыми ошибками ML-инженера (data leakage, train-serve skew).

**Задача:** по 80 признакам дома (площадь, качество материалов, состояние, гараж, бассейн и т.д.) предсказать его стоимость в долларах. Датасет — [Ames Housing](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle).

---

## Содержание

- [Ключевые решения](#ключевые-решения)
- [Результаты и метрики](#результаты-и-метрики)
- [Стадии развития проекта](#стадии-развития-проекта)
- [Структура проекта](#структура-проекта)
- [Как устроен пайплайн](#как-устроен-пайплайн)
- [Запуск проекта](#запуск-проекта)
- [API](#api)
- [Тестирование](#тестирование)
- [Документация](#документация)

---

## Ключевые решения

Полное обоснование каждого решения — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Краткая сводка:

| # | Решение | Суть |
|---|---------|------|
| 1 | **Research ≠ Production** | Jupyter-ноутбуки (`notebooks/`) — только артефакты исследования. Весь финальный код рефакторится в классы и скрипты (`src/`, `api/`). |
| 2 | **Защита от Data Leakage** | Строгий порядок обучения: `train_test_split` → очистка `train` → `preprocessor.fit(train)` → `transform(train/test)`. Очистка строк (аномалии, дубликаты) вынесена за пределы трансформеров признаков, чтобы не ломать размерность `X` и `y`. |
| 3 | **Шаблонный метод вместо спагетти** | Абстрактный `BaseDataPreprocessor` задаёт единый интерфейс `.fit()`/`.transform()` и общую логику (отбор фичей, импутация). `LinearDataPreprocessor` и `CatBoostDataPreprocessor` переопределяют только специфичные методы (OHE + `StandardScaler` vs. работа со строками напрямую). |
| 4 | **Борьба с Train-Serve Skew (умная импутация)** | В API всего **5 обязательных полей** с наибольшей предсказательной силой. Отсутствующие данные обрабатываются по 3 сценариям: логическое отсутствие → `0`; неизвестные числа → медианы из `train`; неизвестные категории → моды из `train`. |
| 5 | **Baseline → Champion** | Baseline — `ElasticNet` с `GridSearchCV` (MAPE ~12.4%). Champion — `CatBoost` "из коробки" без OHE (MAPE ~10.7%), с разделением `eval_metric` (Early Stopping) и `loss_function`. |
| 6 | **Модель и препроцессор — единый артефакт** | В `models/` (в `.gitignore`) всегда сохраняются оба артефакта (`catboost_model.cbm` + `catboost_preprocessor.pkl`), что даёт 100% воспроизводимость логики на бэкенде. |
| 7 | **CI-пайплайн в Docker** | Multistage Dockerfile: стадия `tester` прогоняет `pytest`, стадия `runner` собирает образ только с production-зависимостями. |

---

## Результаты и метрики

| Модель | MAPE (test) | R² (test) | MAE (test) | Примечание |
|--------|-------------|-----------|------------|------------|
| Linear Regression (raw) | ~12.9% | 0.834 | $21 925 | Первый бейзлайн |
| **ElasticNet** (`GridSearchCV`, `alpha`, `l1_ratio`) | **~12.4%** | **0.829** | **$21 531** | **Baseline** — выбран за меньшую ошибку. Слабая L1-регуляризация: почти все признаки полезны |
| **CatBoost** (`iterations=412`, `lr=0.05`, `eval_metric='MAPE'`, `early_stopping_rounds=50`) | **~10.7%** | **0.895** | **$17 621** | **Champion** — превосходит линейную модель, работает со строками напрямую |

Детали экспериментов: `notebooks/linear_models.ipynb`, `notebooks/tree_models.ipynb`.

---

## Стадии развития проекта

История ведётся в Git по GitFlow: фичи — в ветках `feature/*`, вливаются через Pull Request; релиз в `main` — только после готовности бэкенда.

1. **Инициализация и загрузка данных** — структура репозитория, данные Kaggle, описание фичей (`data_description.txt`, перевод `Ames_Housing_Features_Translation.md`).
2. **EDA** (`notebooks/eda.ipynb`) — профилирование данных (`ydata-profiling`), корреляции с таргетом, анализ пропусков и аномалий (дома с `GrLivArea > 4000` и ценой < $200K).
3. **Baseline: линейные модели** (`notebooks/linear_models.ipynb`) — сравнение `LinearRegression`/`ElasticNet`, подбор гиперпараметров `GridSearchCV`. Выбран `ElasticNet` как baseline.
4. **Рефакторинг структуры** — разделение на `src/` (production-код), `notebooks/` (исследования), `tests/` (юнит-тесты).
5. **Класс `DataPreprocessor` и тесты (TDD)** — логика препроцессинга из ноутбука упакована в класс, покрыта `pytest`.
6. **Деревянные модели** — выделен абстрактный `BaseDataPreprocessor` с паттерном «Шаблонный метод», добавлен `CatBoostDataPreprocessor` и пайплайн сравнения CatBoost vs baseline.
7. **Препроцессинг для инференса** — стратегия умной импутации (нули/медианы/моды), обработка отсутствующих полей без train-serve skew, обновлены тесты.
8. **FastAPI-бэкенд** — Pydantic-схемы с 5 обязательными полями, сервис загрузки артефактов, эндпоинты `/health` и `/predict`, тесты API.
9. **Docker** — multistage build (тесты в сборке + лёгкий runtime-образ), зависимости разделены на `requirements.txt` / `requirements-dev.txt`.

---

## Структура проекта

```
ml_pipeline/
├── api/                        # FastAPI-бэкенд
│   ├── main.py                 #   приложение, эндпоинты /health, /predict
│   ├── schemas.py              #   Pydantic-модели запроса/ответа (5 обязательных полей)
│   └── service.py              #   загрузка артефактов, инференс (CatBoost)
├── data/                       # датасет Ames Housing + описание признаков
├── docs/
│   └── ARCHITECTURE.md         # архитектурные и ML-решения проекта
├── models/                     # НЕ в git: обученные артефакты (модель + препроцессор)
│   ├── catboost_model.cbm
│   ├── catboost_preprocessor.pkl
│   ├── linear_preprocessor.pkl
│   └── baseline_model.pkl
├── notebooks/                  # research-артефакты (EDA, эксперименты)
│   ├── eda.ipynb
│   ├── linear_models.ipynb
│   └── tree_models.ipynb
├── src/
│   └── preprocessing.py        # BaseDataPreprocessor + Linear/CatBoost наследники
├── tests/
│   ├── test_api.py             # тесты API (health, predict, валидация 422)
│   └── test_data_preprocessor.py  # юнит-тесты препроцессинга
├── Dockerfile                  # multistage: tester (pytest) → runner (uvicorn)
├── requirements.txt            # production-зависимости
├── requirements-dev.txt        # зависимости для разработки/тестов
└── NOTES.md                    # заметки и best practices по разработке
```

---

## Как устроен пайплайн

```
train.csv
   │
   v
train_test_split ──────────────► test
   │
   v
clean_train_data (дубликаты, аномалии)     ← до трансформеров, чтобы не ломать X и y
   │
   v
preprocessor.fit(train)          препроцессор запоминает медианы/моды из train
   │
   v
transform(train/test)           3 сценария импутации:
   1) логическое отсутствие (гараж, бассейн)  → 0
   2) неизвестные числа                       → медиана (из train)
   3) неизвестные категории                  → мода (из train)
   │
   ├── LinearDataPreprocessor  → OHE + StandardScaler (для Lasso/Ridge/ElasticNet)
   └── CatBoostDataPreprocessor → сырые строки, без скейлинга (инвариантен к масштабу)
   │
   v
model.fit / predict ──────────► артефакты в models/ (модель + препроцессор вместе)
   │
   v
FastAPI (/predict) ───────────► POST: Pydantic-схема → transform → predict → цена в USD
```

---

## Запуск проекта

### Локально

```bash
# 1. Установка зависимостей
pip install -r requirements-dev.txt     # разработка + тесты
# или только production:
pip install -r requirements.txt

# 2. Запуск API (артефакты модели должны лежать в models/)
uvicorn api.main:app --reload

# 3. Открыть Swagger UI: http://localhost:8000/docs
```

### Docker (multistage: тесты → runtime)

```bash
docker build -t ames-housing-api .
docker run -p 8000:8000 ames-housing-api
```

Во время сборки стадия `tester` прогоняет весь `pytest` — образ не соберётся, если тесты красные. В runtime-образ попадают только `src/`, `api/`, `models/` и production-зависимости.

---

## API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка живости сервиса → `{"status": "ok"}` |
| POST | `/predict` | Предсказание стоимости дома |

**Обязательные поля** (валидируются Pydantic): `GrLivArea`, `OverallQual`, `YearBuilt`, `LotArea`, `OverallCond`. Остальные ~30 признаков — `Optional`: сервис сам заполнит их нулём, медианой или модой (по сценариям умной импутации).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1710.0,
    "OverallQual": 7,
    "YearBuilt": 2003,
    "LotArea": 8450.0,
    "OverallCond": 5,
    "GarageArea": 548.0,
    "CentralAir": "Y"
  }'
```

Ответ:

```json
{
  "predicted_price": 214653.792,
  "currency": "USD"
}
```

---

## Тестирование

```bash
pytest tests/ -v
```

- `test_data_preprocessor.py` — режим мод/медиан, отсутствие NaN после `transform`, работа с/без таргета, кодирование `CentralAir`;
- `test_api.py` — health-check, предсказание только по 5 обязательным полям, `422` при пропущенном обязательном поле.

---

## Документация

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — подробное описание архитектурных и ML-решений (leakage, Template Method, умная импутация, выбор моделей, упаковка артефактов).
- [`data/Ames_Housing_Features_Translation.md`](data/Ames_Housing_Features_Translation.md) — перевод и описание признаков датасета.