import pytest
import pandas as pd
import numpy as np
from pynssp import alert_ewma


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


def test_alert_ewma(df, df2):

    df_ewma = alert_ewma(df)
    df2_ewma = alert_ewma(df2.groupby('regions'))

    assert isinstance(df_ewma, pd.DataFrame)
    assert len(df_ewma.columns) == len(df.columns) + 4
    assert isinstance(df2_ewma, pd.DataFrame)
    assert len(df2_ewma.columns) == len(df2.columns) + 4

    with pytest.raises(Exception):
        alert_ewma(df, B=6)

    with pytest.raises(Exception):
        alert_ewma(df, B=785)

    with pytest.raises(Exception):
        alert_ewma(df, g=-1)
