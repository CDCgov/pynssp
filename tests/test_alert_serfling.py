import pytest
import pandas as pd
import numpy as np
from pynssp import alert_serfling


@pytest.fixture
def df():
    return pd.DataFrame({
        'date': pd.date_range('2014-01-05', '2022-02-05', freq='W'),
        'count': np.random.poisson(lam=25, size=422)
    })


@pytest.fixture
def df2():
    return pd.DataFrame({
        'date': pd.date_range('2014-01-05', '2022-02-05', freq='W').repeat(2),
        'regions': np.tile(['reg1', 'reg2'], 422),
        'count': np.floor(np.random.uniform(low=0, high=101, size=422 * 2))
    })


@pytest.fixture
def df3():
    return pd.DataFrame({
        'date': pd.date_range('2014-01-05', '2022-02-05', freq='W').repeat(2),
        'regions': np.tile(['reg1', 'reg2'], 422),
        'count': np.floor(np.random.uniform(low=-100, high=100, size=422 * 2))
    })


def test_alert_serfling(df, df2, df3):

    df_serfling = alert_serfling(df, baseline_end='2020-03-01')
    df2_serfling = alert_serfling(df2.groupby('regions'), baseline_end='2020-03-01')

    assert isinstance(df_serfling, pd.DataFrame)
    assert len(df_serfling.columns) == len(df.columns) + 5
    assert isinstance(df2_serfling, pd.DataFrame)
    assert len(df2_serfling.columns) == len(df2.columns) + 5

    with pytest.raises(Exception):
        alert_serfling(df, baseline_end='2014-01-31')

    with pytest.raises(Exception):
        alert_serfling(df2, baseline_end='2014-01-31')

    with pytest.raises(Exception):
        alert_serfling(df3, baseline_end='2014-01-31')
