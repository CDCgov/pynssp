import pytest
from pynssp import get_api_graph, Credentials, Token, APIGraph
import requests


@pytest.fixture
def url():
    return "http://httpbin.org/image/png"


def test_get_api_graph_credentials(url):

    handle = Credentials(" ", " ")
    graph = handle.get_api_graph(url)

    assert isinstance(graph, APIGraph)
    assert isinstance(graph.response, requests.models.Response)
    assert isinstance(graph.path, str)

    with pytest.raises(Exception):
        get_api_graph(url, profile=[])


def test_get_api_graph_token(url):
    url = "http://httpbin.org/image/png"
    
    handle = Token("abc1234567890")
    
    graph = handle.get_api_graph(url)
    
    assert isinstance(graph, APIGraph)
    assert isinstance(graph.response, requests.models.Response)
    assert isinstance(graph.path, str)

    with pytest.raises(Exception):
        get_api_graph(url, profile=[])
