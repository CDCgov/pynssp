import pytest
from pynssp import (
    create_profile, create_token_profile, create_apikey_profile,
    Credentials, Token, Apikey
)


@pytest.fixture
def myProfile():
    return create_profile("", "")

@pytest.fixture
def myTokenProfile():
    return create_token_profile("", "")

@pytest.fixture
def myApikeyProfile():
    return create_apikey_profile("", "")


def test_create_profile(myProfile):
    assert isinstance(myProfile, Credentials)

    with pytest.raises(Exception):
        create_profile()

    with pytest.raises(Exception):
        create_profile(" ", None)


def test_create_token_profile(myTokenProfile):
    assert isinstance(myTokenProfile, Token)

    with pytest.raises(Exception):
        create_token_profile()


def test_create_apikey_profile(myApikeyProfile):
    assert isinstance(myApikeyProfile, Apikey)

    with pytest.raises(Exception):
        create_apikey_profile()
