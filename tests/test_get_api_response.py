from pynssp import get_api_response, Credentials, Token
import requests


def test_get_api_response():
    url = "http://httpbin.org/json"
    handle = Credentials(" ", " ")
    handle2 = Token("abc1234567890")

    assert isinstance(handle, Credentials)
    assert isinstance(handle2, Token)

    response = get_api_response(url, profile=handle)
    response2 = get_api_response(url, profile=handle2)

    assert isinstance(response, requests.models.Response)
    assert isinstance(response2, requests.models.Response)
