from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class HousePredictRequest(BaseModel):
    # ОБЯЗАТЕЛЬНЫЕ ПОЛЯ
    GrLivArea: float = Field(gt=0,
                             description="Жилая площадь (кв. футы, больше 0)")
    OverallQual: int = Field(ge=1, le=10,
                             description="Общее качество материалов (от 1 до 10)")
    YearBuilt: int = Field(gt=1800, le=2030,
                           description="Год оригинальной постройки")
    LotArea: float = Field(gt=0,
                           description="Площадь участка (кв. футы, больше 0)")
    OverallCond: int = Field(ge=1, le=10,
                             description="Текущее состояние (от 1 до 10)")

    # ОПЦИОНАЛЬНЫЕ ПОЛЯ (Можно не передавать)
    # 1. Поля логического отсутствия (заполним 0, если None)
    GarageArea: Optional[float] = Field(None, ge=0)
    BsmtFullBath: Optional[int] = Field(None, ge=0)
    BsmtHalfBath: Optional[int] = Field(None, ge=0)
    Fireplaces: Optional[int] = Field(None, ge=0)
    PoolArea: Optional[float] = Field(None, ge=0)
    WoodDeckSF: Optional[float] = Field(None, ge=0)
    OpenPorchSF: Optional[float] = Field(None, ge=0)
    EnclosedPorch: Optional[float] = Field(None, ge=0)
    ScreenPorch: Optional[float] = Field(None, ge=0)
    SsnPorch3: Optional[float] = Field(None, alias="3SsnPorch", ge=0)

    # 2. Поля незнания: Числовые (заполним медианой, если None)
    MSSubClass: Optional[int] = None
    FullBath: Optional[int] = None
    LowQualFinSF: Optional[float] = None
    KitchenAbvGr: Optional[int] = None
    YearRemodAdd: Optional[int] = None
    MiscVal: Optional[float] = None

    # 3. Поля незнания: Категориальные (заполним модой, если None)
    Foundation: Optional[str] = None
    BsmtQual: Optional[str] = None
    KitchenQual: Optional[str] = None
    GarageCond: Optional[str] = None
    ExterQual: Optional[str] = None
    GarageQual: Optional[str] = None
    HeatingQC: Optional[str] = None
    ExterCond: Optional[str] = None
    GarageType: Optional[str] = None
    CentralAir: Optional[str] = None
    BsmtCond: Optional[str] = None
    Electrical: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
        # OpenAPI пример для автоматической документации в Swagger UI
        json_schema_extra={
            "example": {
                "GrLivArea": 1710.0,
                "OverallQual": 7,
                "YearBuilt": 2003,
                "LotArea": 8450.0,
                "OverallCond": 5,
                "GarageArea": 548.0,
                "3SsnPorch": 0.0,
                "CentralAir": "Y"
            }
        }
    )


class HousePredictResponse(BaseModel):
    predicted_price: float = Field(description="Предсказание стоимости дома")
    currency: str = "USD"
