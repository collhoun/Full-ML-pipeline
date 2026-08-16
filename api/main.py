from fastapi import FastAPI, HTTPException
from api.schemas import HousePredictRequest, HousePredictResponse
from api.service import model_service
import uvicorn

app = FastAPI(
    title="Ames Housing prediction api",
    description="API для предсказание цены на дома с помощью CatBoost"
)


@app.get("/health", tags="System")
def health_check():
    return {'status': 'ok'}


@app.post("/predict", response_model=HousePredictResponse, tags="Prediction")
def predict_price(request: HousePredictRequest):
    """Принимает параметры дома и возвращает оценку стоимости"""
    try:
        price = model_service.predict(request)
        return HousePredictResponse(predicted_price=price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
