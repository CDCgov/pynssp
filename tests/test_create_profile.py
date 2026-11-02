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
    return create_apikey_profile("", "API-KEY")


def test_create_profile(myProfile, monkeypatch):
    assert isinstance(myProfile, Credentials)

    monkeypatch.setattr("builtins.input", lambda *_: "user")
    monkeypatch.setattr("pynssp.utils.getpass", lambda *_, **__: "pass")
    prompted_profile = create_profile()
    assert isinstance(prompted_profile, Credentials)

    monkeypatch.setattr("pynssp.utils.getpass", lambda *_, **__: "")
    blank_password_profile = create_profile(" ", None)
    assert isinstance(blank_password_profile, Credentials)


def test_create_token_profile(myTokenProfile, monkeypatch):
    assert isinstance(myTokenProfile, Token)

    monkeypatch.setattr("pynssp.utils.getpass", lambda *_, **__: "token")
    prompted_profile = create_token_profile()
    assert isinstance(prompted_profile, Token)


def test_create_apikey_profile(myApikeyProfile, monkeypatch):
    assert isinstance(myApikeyProfile, Apikey)

    monkeypatch.setattr("pynssp.utils.getpass", lambda *_, **__: "apikey")
    prompted_profile = create_apikey_profile()
    assert isinstance(prompted_profile, Apikey)
