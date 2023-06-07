import pytest
from pynssp import get_api_graph, Credentials, Token, APIGraph
import requests


def test_get_api_graph():
    url = "http://httpbin.org/image/png"

    graph = get_api_graph(url, profile=handle)
    graph2 = get_api_graph(url, profile=handle2)

    assert isinstance(graph, APIGraph)
    assert isinstance(graph.response, requests.models.Response)
    assert isinstance(graph.path, str)

    with pytest.raises(Exception):
        get_api_graph(url, profile=[])