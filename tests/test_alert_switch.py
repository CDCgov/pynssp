import pytest
import pandas as pd
import numpy as np
from pynssp import alert_switch


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


def test_alert_switch(df, df2):

    df_switch = alert_switch(df)
    df2_switch = alert_switch(df2.groupby('regions'))

    assert isinstance(df_switch, pd.DataFrame)
    assert len(df_switch.columns) == len(df.columns) + 5
    assert isinstance(df2_switch, pd.DataFrame)
    assert len(df2_switch.columns) == len(df2.columns) + 5

    with pytest.raises(Exception):
        alert_switch(df, B=6)

    with pytest.raises(Exception):
        alert_switch(df, B=15)

    with pytest.raises(Exception):
        alert_switch(df, B=784)

    with pytest.raises(Exception):
        alert_switch(df, g=-1)
