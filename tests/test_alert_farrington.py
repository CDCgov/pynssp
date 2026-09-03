import pytest
import pandas as pd
import numpy as np
from pynssp import alert_farrington


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
        'date': pd.date_range('2020-01-01', '2020-12-31'),
        'count': np.floor(np.random.uniform(low=-100, high=100, size=366))
    })


def test_alert_farrington(df, df2, df3):

    df_farr_original = alert_farrington(df)
    df_farr_original = alert_farrington(df2.groupby('regions'))

    df2_farr_modified = alert_farrington(df, method='modified')
    df2_farr_modified = alert_farrington(df2.groupby('regions'), method='modified')

    assert isinstance(df_farr_original, pd.DataFrame)
    assert isinstance(df_farr_original, pd.DataFrame)
    assert isinstance(df2_farr_modified, pd.DataFrame)
    assert isinstance(df2_farr_modified, pd.DataFrame)
    assert len(df_farr_original.columns) == len(df.columns) + 6
    assert len(df_farr_original.columns) == len(df2.columns) + 6
    assert len(df2_farr_modified.columns) == len(df.columns) + 6
    assert len(df2_farr_modified.columns) == len(df2.columns) + 6

    with pytest.raises(Exception):
        alert_farrington(df, method="notamethod")

    with pytest.raises(Exception):
        alert_farrington(df2, method="notamethod")

    with pytest.raises(Exception):
        alert_farrington(df3)

    with pytest.raises(Exception):
            alert_farrington(df3, method='modified')
