import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataPreprocessor:

    numeric_columns = ['BsmtFullBath', 'WoodDeckSF', 'Fireplaces', '3SsnPorch', 'ScreenPorch', 'MSSubClass', 'LotArea', 'GrLivArea', 'FullBath',
                       'PoolArea', 'YearBuilt', 'LowQualFinSF', 'OpenPorchSF', 'OverallQual', 'GarageArea', 'BsmtHalfBath', 'YearRemodAdd', 'EnclosedPorch', 'OverallCond', 'KitchenAbvGr', 'MiscVal']
    category_columns = ['Foundation', 'BsmtQual', 'KitchenQual', 'GarageCond', 'ExterQual',
                        'GarageQual', 'HeatingQC', 'ExterCond', 'GarageType', 'CentralAir', 'BsmtCond', 'Electrical']
    features = numeric_columns + category_columns
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    ordinal_columns = ['ExterQual', 'HeatingQC', 'GarageCond',
                       'KitchenQual', 'ExterCond', 'BsmtQual', 'GarageQual', 'BsmtCond']
    target_column = 'SalePrice'
    one_hots = ['Foundation', 'GarageType', 'Electrical']

    def __init__(self, drop: None | str = None) -> None:
        self.scaler = StandardScaler()
        self.electrical_mode = None
        self.ohe_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, drop=drop)

    def fit(self, df: pd.DataFrame):
        """
        Вычисляет статистики по тренировочной выборке после категориального кодирования:
        - вычисляет моду по фиче electrical
        - вычисляет среднее и дисперсии для standartscaler

        Args:
            df (pd.DataFrame): train df
        """
        self.electrical_mode = df['Electrical'].mode()[
            0]
        df_filled = df.copy()
        df_filled['Electrical'] = df_filled['Electrical'].fillna(
            self.electrical_mode)
        df_filled['GarageType'] = df_filled['GarageType'].fillna('NoGarage')
        self.ohe_encoder.fit(df_filled[self.one_hots])

        df_encoded = self.feature_selection(df)
        df_encoded = self.code_categories(df_encoded)
        df_encoded = self.fill_missing_with_zeroes(df_encoded)
        scale_columns = self._get_scale_columns()
        self.scaler.fit(df_encoded[scale_columns])
        return self

    def code_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        кодирует категориальные переменные с помощью qual map и OHE encoder, который обучается в методе "fit"

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: данные с закодированными категориальными переменными
        """
        df = df.copy()
        df['CentralAir'] = df['CentralAir'].apply(
            lambda x: 1 if x == 'Y' else 0)
        for column in self.ordinal_columns:
            df[column] = df[column].map(self.qual_map)
            df[column] = df[column].fillna(0)

        df['GarageType'] = df['GarageType'].fillna('NoGarage')
        df['Electrical'] = df['Electrical'].fillna(self.electrical_mode)

        encoded = self.ohe_encoder.transform(df[self.one_hots])
        encoded_cols = self.ohe_encoder.get_feature_names_out(self.one_hots)
        encoded_df = pd.DataFrame(
            encoded, columns=encoded_cols, index=df.index)

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

    def delete_dublicated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        удаляет дубликаты НА ТРЕНИРОВОЧНОЙ ВЫБОРКЕ

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: данные с удаленными дубликатами
        """
        return df.drop_duplicates()

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
            (df['GrLivArea'] > 4000) &
            (df['SalePrice'] < 200_000)
        ].index
        return df.drop(index=anomaly_indices)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        изменяет данные и возвращает датафрейм после препроцессинга

        Args:
            df (pd.DataFrame): выборка данных

        Returns:
            pd.DataFrame: датафрейм после препроцессинга
        """
        df = self.feature_selection(df)
        df = self.code_categories(df)
        df = self.fill_missing_with_zeroes(df)

        scale_columns = self._get_scale_columns()
        df[scale_columns] = self.scaler.transform(df[scale_columns])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

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
