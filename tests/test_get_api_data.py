from pynssp import get_api_data, Credentials, Token
import pandas as pd


def test_get_api_data():
    url = "http://httpbin.org/json"
    url2 = "http://httpbin.org/robots.txt"

    handle = Credentials(" ", " ")
    handle2 = Token("abc1234567890")

    assert isinstance(handle, Credentials)
    assert isinstance(handle2, Token)

    data = get_api_data(url, profile=handle)
    data2 = get_api_data(url2, profile=handle, fromCSV=True)
    data3 = get_api_data(url, profile=handle2)
    data4 = get_api_data(url2, profile=handle2, fromCSV=True)

    assert isinstance(data, pd.DataFrame)
    assert isinstance(data2, pd.DataFrame)
    assert isinstance(data3, pd.DataFrame)
    assert isinstance(data4, pd.DataFrame)