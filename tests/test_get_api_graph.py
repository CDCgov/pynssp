import pytest
from pynssp import get_api_graph, Credentials, Token, APIGraph
import requests


def test_get_api_graph():
    url = "http://httpbin.org/image/png"

    handle = Credentials(" ", " ")
    handle2 = Token("abc1234567890")

    graph = get_api_graph(url, profile=handle)
    graph2 = handle.get_api_graph(url)
    graph3 = get_api_graph(url, profile=handle2)
    graph4 = handle2.get_api_graph(url)

    assert isinstance(graph, APIGraph)
    assert isinstance(graph2, APIGraph)
    assert isinstance(graph3, APIGraph)
    assert isinstance(graph4, APIGraph)

    assert isinstance(graph.response, requests.models.Response)
    assert isinstance(graph2.response, requests.models.Response)
    assert isinstance(graph3.response, requests.models.Response)
    assert isinstance(graph4.response, requests.models.Response)

    assert isinstance(graph.path, str)
    assert isinstance(graph2.path, str)
    assert isinstance(graph3.path, str)
    assert isinstance(graph4.path, str)

    with pytest.raises(Exception):
        get_api_graph(url, profile=[])