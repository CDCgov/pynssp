import os
import pytest
from pynssp import get_api_response, Credentials, Token
import requests


@pytest.fixture
def profile():
    return Credentials(" ", " ")


@pytest.fixture
def tokenProfile():
    return Token("abc1234567890")


def test_profile_class(profile, tokenProfile):
    assert isinstance(profile, Credentials)
    assert isinstance(tokenProfile, Token)


def test_profile_save(profile, tokenProfile):
    profile.pickle()
    tokenProfile.pickle()
    profile.pickle(file="test.pkl")
    tokenProfile.pickle(file="test2.pkl")

    assert os.path.isfile("myProfile.pkl")
    assert os.path.isfile("tokenProfile.pkl")
    assert os.path.isfile("test.pkl")
    assert os.path.isfile("test2.pkl")

    test_files = os.listdir(".")

    for test_file in test_files:
        if test_file.endswith(".pkl"):
            os.remove(os.path.join(".", test_file))


def test_get_api_response():
    url = "http://httpbin.org/json"

    handle = Credentials(" ", " ")
    handle2 = Token("abc1234567890")

    response = get_api_response(url, profile=handle)
    response2 = get_api_response(url, profile=handle2)

    assert isinstance(response, requests.models.Response)
    assert isinstance(response2, requests.models.Response)

    with pytest.raises(Exception):
        get_api_response(url, profile=[])
