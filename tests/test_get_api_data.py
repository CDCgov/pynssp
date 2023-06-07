import os
import pytest
from pynssp import get_api_data, Credentials, Token
import pandas as pd


@pytest.fixture
def url():
    return "http://httpbin.org/json"


@pytest.fixture
def url2():
    return "http://httpbin.org/robots.txt"


def test_get_api_data_credentials(url, url2):

    handle = Credentials(" ", " ")
    data = get_api_data(url, profile=handle)
    data2 = get_api_data(url2, profile=handle, fromCSV=True)

    assert isinstance(data, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)

    with pytest.raises(Exception):
        get_api_data(url, profile=None)


def test_get_api_data_token(url, url2):
    
    handle = Token("abc1234567890")
    data = get_api_data(url, profile=handle)
    data2 = get_api_data(url2, profile=handle, fromCSV=True)
    
    assert isinstance(data, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)

    with pytest.raises(Exception):
        get_api_data(url, profile=None)