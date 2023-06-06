import pytest
import pandas as pd
import numpy as np
from pynssp import classify_trend


@pytest.fixture
def df():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31'),
        'dataCount': np.floor(np.random.uniform(low=0, high=101, size=366)),
        'allCount': np.floor(np.random.uniform(low=101, high=500, size=366))
    })


@pytest.fixture
def df2():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31').repeat(2),
        'dataCount': np.floor(np.random.uniform(low=0, high=101, size=366 * 2)),
        'allCount': np.floor(np.random.uniform(low=101, high=500, size=366 * 2)),
        'regions': np.tile(['reg1', 'reg2'], 366)
    })


def test_classify_trend(df, df2):
    df_trend = classify_trend(df)
    df2_trend = classify_trend(df2.groupby('regions'))

    assert isinstance(df_trend, pd.DataFrame)
    assert len(df_trend.columns) == len(df.columns) + 3
    assert isinstance(df2_trend, pd.DataFrame)
    assert len(df2_trend.columns) == len(df2.columns) + 3