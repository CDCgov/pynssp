import pytest
import pandas as pd
import numpy as np
from pynssp import alert_regression


@pytest.fixture
def df():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31'),
        'count': np.floor(np.random.uniform(low=0, high=101, size=366))
    })


@pytest.fixture
def df2():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31').repeat(2),
        'regions': np.tile(['reg1', 'reg2'], 366),
        'count': np.floor(np.random.uniform(low=0, high=101, size=366 * 2))
    })


def test_alert_regression(df, df2):

    df_regression = alert_regression(df)
    df2_regression = alert_regression(df2.groupby('regions'))

    assert isinstance(df_regression, pd.DataFrame)
    assert len(df_regression.columns) == len(df.columns) + 9
    assert isinstance(df2_regression, pd.DataFrame)
    assert len(df2_regression.columns) == len(df2.columns) + 8

    with pytest.raises(Exception):
        alert_regression(df, B=6)

    with pytest.raises(Exception):
        alert_regression(df, B=15)

    with pytest.raises(Exception):
        alert_regression(df, B=784)

    with pytest.raises(Exception):
        alert_regression(df, g=-1)

    with pytest.raises(Exception):
        alert_regression(df2)
