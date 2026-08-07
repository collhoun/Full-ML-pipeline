import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from abc import ABC, abstractmethod


class BaseDataPreprocessor(ABC):
    """Абстрактный базовый класс для препроцессинга данных Ames Housing.

    Содержит общую логику очистки, выбора фичей и заполнения пропусков.
    Определяет шаблон работы методов fit, transform и fit_transform.
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
        """Общая базовая импутация категорий до закодирования

        Args:
            df (pd.DataFrame): входная выборка

        Returns:
            pd.DataFrame: обще закодированная выборка
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
        self.electrical_mode = df['Electrical'].mode()[0]

        # Вызываем специфичное для дочернего класса обучение кодировщиков/скейлеров
        df_clean = self.feature_selection(df)
        df_clean = self._base_category_imputation(df_clean)
        df_clean = self.fill_missing_with_zeroes(df_clean)
        self._fit_specific(df_clean)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = self.feature_selection(df)
        df_processed = self._base_category_imputation(df_processed)
        df_processed = self.code_categories_with_ohe(df_processed)
        df_processed = self.fill_missing_with_zeroes(df_processed)
        df_processed = self._scale_specific(df_processed)
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @abstractmethod
    def code_categories_with_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
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
    """Класс для препроцессора признаков домов из набора данных Ames Housing.

    Основное назначение:
    - выбор нужных колонок по заданным спискам признаков
    - кодирование категориальных признаков
    - заполнение пропусков нулями (кроме целевой колонки)
    - масштабирование числовых признаков
    - обучение трансформеров на тренировочных данных

    Этот класс готовит матрицу признаков для моделей, ожидающих числовой вход.
    Он не должен сам управлять разбиением на X и y, но может выполнять очистку строк
    в методе `clean_train_data` для тренировочного набора.

    Параметры:
        drop: None | str
            Передается в `OneHotEncoder(drop=drop)`.
            При `drop='first'` сохраняется OHE без одной из категорий.
            При `drop=None` сохраняются все бинарные колонки.

    Важные ограничения:
    - метод `clean_train_data` удаляет строки, поэтому его лучше вызывать до выделения X и y.
    - метод `feature_selection` ожидает наличие всех признаков из `self.features`.
    - `fill_missing_with_zeroes` не заполняет пропуски в `SalePrice`.
    - `OneHotEncoder(handle_unknown='ignore')` безопасно обрабатывает новые категории на инференсе,
      но для них не создаются новые колонки.

    Методы:
    - fit(df): обучает OHE и StandardScaler на тренировочных данных.
    - transform(df): применяет кодирование и масштабирование к новым данным.
    - fit_transform(df): вызывает `fit` и сразу `transform`.
    - feature_selection(df): выбирает нужные признаки и, при наличии, целевой столбец.
    - code_categories_with_ohe(df): кодирует `CentralAir`, ordinal-признаки и применяет OHE к `one_hots`.
    - _get_scale_columns(): возвращает колонки для масштабирования.
    - fill_missing_with_zeroes(df): заполняет пропуски нулями по признакам.
    - delete_dublicated(df): удаляет дубликаты строк.
    - delete_anomalies(df): удаляет аномалии по `GrLivArea` и `SalePrice`.
    - clean_train_data(df): объединяет удаление дубликатов и аномалий.
    """
    one_hots = ['Foundation', 'GarageType', 'Electrical']

    def __init__(self, drop: None | str = None) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.ohe_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, drop=drop)

    def code_categories_with_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df_encoded = self.code_categories_with_ohe(df)
        scale_columns = self._get_scale_columns()
        self.scaler.fit(df_encoded[scale_columns])

    def _scale_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученный scaler к scale_columns.

        Args:
            df (pd.DataFrame): данные после code_categories_with_ohe и fill_missing_with_zeroes

        Returns:
            pd.DataFrame: датафрейм с отмасштабированными колонками
        """
        df = df.copy()
        scale_columns = self._get_scale_columns()
        df[scale_columns] = self.scaler.transform(df[scale_columns])
        return df
