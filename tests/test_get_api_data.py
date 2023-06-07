import pytest
from pynssp import get_api_data, Credentials, Token
import pandas as pd


def test_get_api_data():
    url = "http://httpbin.org/json"
    url2 = "http://httpbin.org/robots.txt"

    assert isinstance(handle, Credentials)
    assert isinstance(handle2, Token)

    data = get_api_data(url, profile=handle)
    data2 = get_api_data(url2, profile=handle, fromCSV=True)

    assert isinstance(data, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)
