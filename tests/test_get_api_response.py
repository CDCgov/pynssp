import os
import pytest
from pynssp import get_api_response, Credentials, Token, Apikey
import requests


@pytest.fixture
def profile():
    return Credentials(" ", " ")


@pytest.fixture
def tokenProfile():
    return Token("abc1234567890")

@pytest.fixture
def apikeyProfile():
    return Apikey(" ", " ")


def test_profile_class(profile, tokenProfile, apikeyProfile):
    assert isinstance(profile, Credentials)
    assert isinstance(tokenProfile, Token)
    assert isinstance(apikeyProfile, Apikey)


def test_profile_save(profile, tokenProfile, apikeyProfile):
    profile.pickle()
    tokenProfile.pickle()
    apikeyProfile.pickle()
    profile.pickle(file="test.pkl")
    tokenProfile.pickle(file="test2.pkl")
    apikeyProfile.pickle(file="test3.pkl")

    assert os.path.isfile("myProfile.pkl")
    assert os.path.isfile("tokenProfile.pkl")
    assert os.path.isfile("apikeyProfile.pkl")
    assert os.path.isfile("test.pkl")
    assert os.path.isfile("test2.pkl")
    assert os.path.isfile("test3.pkl")

    test_files = os.listdir(".")

    for test_file in test_files:
        if test_file.endswith(".pkl"):
            os.remove(os.path.join(".", test_file))


def test_get_api_response():
    url = "http://httpbin.org/json"

    handle = Credentials(" ", " ")
    handle2 = Token("abc1234567890")
    handle3 = Apikey(" ", " ")

    response = get_api_response(url, profile=handle)
    response2 = get_api_response(url, profile=handle2)
    response3 = get_api_response(url, profile=handle3)

    assert isinstance(response, requests.models.Response)
    assert isinstance(response2, requests.models.Response)
    assert isinstance(response3, requests.models.Response)

    with pytest.raises(Exception):
        get_api_response(url, profile=[])
