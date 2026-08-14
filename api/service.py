import joblib
from catboost import CatBoostRegressor
import pandas as pd
from api.schemas import HousePredictRequest
from pathlib import Path


class HousePriceModelService:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        preprocessor_path = base_dir / "models" / "catboost_preprocessor.pkl"
        model_path = base_dir / "models" / "catboost_model.cbm"

        # загружаем артефакты в память один раз
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))

    def predict(self, request_data: HousePredictRequest) -> float:
        """Принимает Pydantic схему, возвращает предсказание (число)"""

        data_dict = request_data.model_dump(by_alias=True)
        df = pd.DataFrame([data_dict])

        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)[0]

        return max(0.0, float(prediction))


model_service = HousePriceModelService()
