import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from abc import ABC, abstractmethod


class BaseDataPreprocessor(ABC):
    """Абстрактный базовый класс для препроцессинга данных Ames Housing.

    Обеспечивает общий pipeline подготовки признаков: выбор нужных колонок,
    базовую импутацию категориальных признаков, заполнение пропусков нулями
    и шаблон обучения/применения трансформеров. Дочерние классы отвечают за
    специфичное кодирование категориальных признаков и масштабирование.
    """

    numeric_columns = [
        'BsmtFullBath', 'WoodDeckSF', 'Fireplaces', '3SsnPorch', 'ScreenPorch', 'MSSubClass',
        'LotArea', 'GrLivArea', 'FullBath', 'PoolArea', 'YearBuilt', 'LowQualFinSF',
        'OpenPorchSF', 'OverallQual', 'GarageArea', 'BsmtHalfBath', 'YearRemodAdd',
        'EnclosedPorch', 'OverallCond', 'KitchenAbvGr', 'MiscVal'
    ]
    category_columns = [
        'Foundation', 'BsmtQual', 'KitchenQual', 'GarageCond', 'ExterQual',
        'GarageQual', 'HeatingQC', 'ExterCond', 'GarageType', 'CentralAir', 'BsmtCond', 'Electrical'
    ]
    features = numeric_columns + category_columns
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    ordinal_columns = [
        'ExterQual', 'HeatingQC', 'GarageCond', 'KitchenQual',
        'ExterCond', 'BsmtQual', 'GarageQual', 'BsmtCond'
    ]
    target_column = 'SalePrice'

    def __init__(self) -> None:
        self.electrical_mode = None

    def delete_dublicated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        удаляет дубликаты НА ТРЕНИРОВОЧНОЙ ВЫБОРКЕ

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: данные с удаленными дубликатами
        """
        return df.drop_duplicates()

    def delete_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        удаляет аномалии из ТРЕНИРОВОЧНОЙ ВЫБОРКЕ. вызывается до feature_selection
        аномалиями считаются дома с GrLivArea > 4000 и стоимостью меньше 200_000 долларов

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: датафрейм без аномалий
        """
        df = df.copy()
        anomaly_indices = df[
            (df['GrLivArea'] > 4000) & (df['SalePrice'] < 200_000)
        ].index
        return df.drop(index=anomaly_indices)

    def clean_train_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        очищащет тренировочную выборку от дубликатов и аномалий. вызывается до feature_selection

        Args:
            df (pd.DataFrame): тренировочная выборка

        Returns:
            pd.DataFrame: очищенная тренировочная выборка
        """
        df = self.delete_dublicated(df)
        df = self.delete_anomalies(df)
        return df

    def feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Отбирает нужные фичи среди остальных

        Args:
            df (pd.DataFrame): исходная выборка

        Returns:
            pd.DataFrame: урезанная выборка
        """
        cols_to_keep = self.features.copy()
        if self.target_column in df.columns:
            cols_to_keep.append(self.target_column)
        return df[cols_to_keep].copy()

    def fill_missing_with_zeroes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        заполняет пропуски нулями

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: данные с заполненными пропусками
        """
        df = df.copy()
        missing_info = df.isnull().sum()
        missing_cols = missing_info[(missing_info > 0) & (
            missing_info.index != self.target_column)]
        df[missing_cols.index] = df[missing_cols.index].fillna(0)
        return df

    def _base_category_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выполняет базовую обработку категориальных признаков перед кодированием.

        Преобразует колонку CentralAir в бинарный формат, кодирует ordinal-поля
        по словарю qual_map, заполняет GarageType значением NoGarage и
        подставляет наиболее частое значение Electrical, рассчитанное на этапе fit.

        Args:
            df (pd.DataFrame): входная выборка

        Returns:
            pd.DataFrame: подготовленная выборка с базово закодированными признаками
        """
        df = df.copy()
        df['CentralAir'] = df['CentralAir'].apply(
            lambda x: 1 if x == 'Y' else 0)
        for column in self.ordinal_columns:
            df[column] = df[column].map(self.qual_map).fillna(0)
        df['GarageType'] = df['GarageType'].fillna('NoGarage')
        if self.electrical_mode is not None:
            df['Electrical'] = df['Electrical'].fillna(self.electrical_mode)
        return df

    # --- ШАБЛОННЫЕ ПУБЛИЧНЫЕ МЕТОДЫ (Единый API) ---

    def fit(self, df: pd.DataFrame):
        """Обучает параметры препроцессора на тренировочных данных.

        На этапе fit вычисляется наиболее частое значение Electrical,
        затем выполняется отбор признаков, базовая импутация категорий,
        заполнение пропусков и обучение специфичных трансформеров.
        """
        self.electrical_mode = df['Electrical'].mode()[0]

        df_clean = self.feature_selection(df)
        df_clean = self._base_category_imputation(df_clean)
        df_clean = self.fill_missing_with_zeroes(df_clean)
        self._fit_specific(df_clean)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет обученные преобразования к новым данным.

        Выполняет отбор признаков, базовую обработку категориальных колонок,
        кодирование категорий, заполнение пропусков и масштабирование.
        """
        df_processed = self.feature_selection(df)
        df_processed = self._base_category_imputation(df_processed)
        df_processed = self._encode_categories_specific(df_processed)
        df_processed = self.fill_missing_with_zeroes(df_processed)
        df_processed = self._scale_specific(df_processed)
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сначала обучает трансформеры, затем сразу применяет их к данным."""
        return self.fit(df).transform(df)

    @abstractmethod
    def _encode_categories_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных переменных (OHE, Ordinal и т.д.)"""
        pass

    @abstractmethod
    def _fit_specific(self, df: pd.DataFrame) -> None:
        """Обучение специфичных трансформеров (OHE, Scaler, OrdinalEncoder)"""
        pass

    @abstractmethod
    def _scale_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение скейлинга (если требуется)"""
        pass


class LinearDataPreprocessor(BaseDataPreprocessor):
    """Препроцессор для линейных моделей на данных Ames Housing.

    Подготавливает признаки для моделей, работающих с числовым входом:
    отбирает нужные колонки, кодирует базовые категориальные признаки,
    применяет one-hot encoding к колонкам Foundation, GarageType и Electrical,
    масштабирует числовые, ordinal и one-hot признаки с помощью StandardScaler.

    Параметры:
        drop: None | str
            Передается в OneHotEncoder(drop=drop). При drop='first' одна из
            категорий исключается из матрицы признаков, при drop=None
            сохраняются все бинарные колонки.

    Важные особенности:
    - для заполнения пропусков в Electrical используется наиболее частое
      значение, вычисленное на этапе fit;
    - метод clean_train_data следует вызывать до разбиения на X и y,
      потому что он удаляет строки из обучающей выборки;
    - OneHotEncoder(handle_unknown='ignore') безопасно обрабатывает новые
      категории на инференсе, но не создает для них новые колонки.
    """
    one_hots = ['Foundation', 'GarageType', 'Electrical']

    def __init__(self, drop: None | str = None) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.ohe_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, drop=drop)

    def _encode_categories_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет OneHotEncoder к one_hots-колонкам.
        Ordinal-признаки и CentralAir кодируются раньше, в _base_category_imputation.

        Args:
            df (pd.DataFrame): выборка данных, уже прошедшая _base_category_imputation

        Returns:
            pd.DataFrame: данные с one-hot закодированными переменными
        """
        df = df.copy()
        encoded = self.ohe_encoder.transform(df[self.one_hots])
        encoded_cols = self.ohe_encoder.get_feature_names_out(self.one_hots)
        encoded_df = pd.DataFrame(
            encoded, columns=encoded_cols, index=df.index)  # type: ignore

        df = df.drop(columns=self.one_hots)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def _get_scale_columns(self) -> list[str]:
        """
        Возвращает полный список колонок, подлежащих масштабированию:
        numeric + ordinal + все one-hot колонки, полученные из encoder'а.
        Вызывать только после того, как self.ohe_encoder уже зафичен.
        """
        ohe_columns = list(
            self.ohe_encoder.get_feature_names_out(self.one_hots))
        return self.numeric_columns + self.ordinal_columns + ohe_columns + ['CentralAir']

    def _fit_specific(self, df: pd.DataFrame) -> None:
        """
        Обучает ohe_encoder и scaler на данных, уже прошедших
        feature_selection + _base_category_imputation (см. base.fit).

        Args:
            df (pd.DataFrame): импутированный train df
        """
        self.ohe_encoder.fit(df[self.one_hots])

        df_encoded = self._encode_categories_specific(df)
        scale_columns = self._get_scale_columns()
        self.scaler.fit(df_encoded[scale_columns])

    def _scale_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученный scaler к scale_columns.

        Args:
            df (pd.DataFrame): данные после _encode_categories_specific и fill_missing_with_zeroes

        Returns:
            pd.DataFrame: датафрейм с отмасштабированными колонками
        """
        df = df.copy()
        scale_columns = self._get_scale_columns()
        df[scale_columns] = self.scaler.transform(df[scale_columns])
        return df


class CatBoostDataPreprocessor(BaseDataPreprocessor):
    """..."""

    cat_string_columns = ['Foundation', 'GarageType', 'Electrical']

    def __init__(self) -> None:
        super().__init__()

    def _encode_categories_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает строковые категориальные признаки для CatBoost.
        Вместо OHE мы просто убеждаемся, что они имеют тип str и не содержат NaN.
        """
        df = df.copy()
        for col in self.cat_string_columns:
            if col in df.columns:
                # как я понял catboost не любит float(NaN), поэтому заполняем Missing
                df[col] = df[col].fillna('Missing').astype(str)
        return df

    def _fit_specific(self, df: DataFrame) -> None:
        """
        Для CatBoost не требуется специфичного обучения (fit) на этом этапе.
        Алгоритм сам разберется со строками во время обучения модели.
        """
        pass

    def _scale_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        деревянные модели инвариантны к масштабу признаков.
        Скейлинг не требуется.
        """
        return df
