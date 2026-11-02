import pytest
from pynssp import get_essence_data, Credentials, Token, Apikey


@pytest.fixture
def myProfile():
    return Credentials(" ", " ")


@pytest.fixture
def tokenProfile():
    return Token("abc1234567890")

@pytest.fixture
def apikeyProfile():
    return Apikey("abcd1234", "API-KEY")


def test_get_essence_data(myProfile, tokenProfile, apikeyProfile):
    url = "http://httpbin.org/json"

    with pytest.raises(Exception):
        get_essence_data(url, start_date="2021-02-15", end_date="2021-02-15", profile=myProfile)

    with pytest.raises(Exception):
        get_essence_data(url, profile=myProfile)

    with pytest.raises(Exception):
        get_essence_data(url, profile=tokenProfile)

    with pytest.raises(Exception):
        get_essence_data(url, profile=apikeyProfile)

    with pytest.raises(Exception):
        get_essence_data(url, profile=[])

    with pytest.raises(Exception):
        get_essence_data(url, profile=None)
