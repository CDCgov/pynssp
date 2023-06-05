import pytest
from pynssp import create_profile, create_token_profile, Credentials, Token


@pytest.fixture
def myProfile():
    return create_profile("", "")

@pytest.fixture
def myTokenProfile():
    return create_token_profile("", "")


def test_create_profile(myProfile):
    assert isinstance(myProfile, Credentials)


def test_create_token_profile(myTokenProfile):
    assert isinstance(myTokenProfile, Token)
