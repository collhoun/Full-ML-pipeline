import pytest
import numpy as np
import pandas as pd
from src.preprocessing import LinearDataPreprocessor


@pytest.fixture
def generate_train_data():
    """Фикстура, которая генерирует фейковый тренировочный датасет"""
    preprocessor = LinearDataPreprocessor()
    df = pd.DataFrame(columns=preprocessor.features +
                      [preprocessor.target_column])

    df['LotArea'] = [1000, 2000, np.nan]
    df['CentralAir'] = ['Y', 'N', 'Y']
    df['ExterQual'] = ['Ex', 'TA', 'Po']  # ordinals
    df['Electrical'] = ['SBrkr', np.nan, 'SBrkr']  # для моды
    df['Foundation'] = ['PConc', 'CBlock', 'BrkTil']  # Для OneHot
    df['SalePrice'] = [200000, 150000, 300000]  # Таргет

    for col in preprocessor.numeric_columns:
        if df[col].isna().all():
            df[col] = 1.0

    for col in preprocessor.category_columns:
        if df[col].isna().all():
            df[col] = 'TA'

    return df


@pytest.fixture
def fitted_preprocessor(generate_train_data):
    """Фикстура: возвращает обученный препроцессор"""
    preprocessor = LinearDataPreprocessor()
    preprocessor.fit(generate_train_data)
    return preprocessor


def test_calcualte_electrical_mode(generate_train_data):
    preprocessor = LinearDataPreprocessor()
    preprocessor.fit(generate_train_data)
    assert preprocessor.electrical_mode == 'SBrkr'


def test_transform_remove_all_nans(fitted_preprocessor, generate_train_data):
    transformed_df = fitted_preprocessor.transform(generate_train_data)
    assert transformed_df.isna().sum().sum() == 0


def test_transform_preserves_target_if_present(fitted_preprocessor, generate_train_data):
    transformed_df = fitted_preprocessor.transform(generate_train_data)
    assert 'SalePrice' in transformed_df.columns
    assert list(transformed_df['SalePrice']) == list(
        generate_train_data['SalePrice'])


def test_transform_works_without_target(fitted_preprocessor, generate_train_data):
    inference_data = generate_train_data.drop(columns=['SalePrice'])
    # если здесь будет ошибка, тест упадет
    transformed_df = fitted_preprocessor.transform(inference_data)
    assert 'SalePrice' not in transformed_df.columns


def test_central_air_logic(fitted_preprocessor, generate_train_data):
    df_selected = fitted_preprocessor.feature_selection(generate_train_data)
    encoded_df = fitted_preprocessor._base_category_imputation(df_selected)
    assert encoded_df['CentralAir'].iloc[0] == 1
    assert encoded_df['CentralAir'].iloc[1] == 0
