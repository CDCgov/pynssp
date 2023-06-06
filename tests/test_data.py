import pytest
import pandas as pd
from pynssp import load_simulated_ts, get_scenario1, get_scenario2, load_nssp_stopwords


@pytest.fixture
def simulated_ts():
    return load_simulated_ts()


@pytest.fixture
def scenario1():
    return get_scenario1()


@pytest.fixture
def scenario2():
    return get_scenario2()


@pytest.fixture
def nssp_stopwords():
    return load_nssp_stopwords()


def test_load_simulated_ts(simulated_ts):
    assert isinstance(simulated_ts, pd.DataFrame)
    assert len(simulated_ts) == 626
    assert len(simulated_ts.columns) == 6


def test_get_scenario1(scenario1):
    assert isinstance(scenario1, pd.DataFrame)
    assert len(scenario1) == 313
    assert len(scenario1.columns) == 6


def test_get_scenario2(scenario2):
    assert isinstance(scenario2, pd.DataFrame)
    assert len(scenario2) == 313
    assert len(scenario2.columns) == 6


def test_load_nssp_stopwords(nssp_stopwords):
    assert isinstance(nssp_stopwords, pd.DataFrame)
    assert len(nssp_stopwords) == 835
    assert len(nssp_stopwords.columns) == 2
