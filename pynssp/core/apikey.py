from requests import get
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer
from pynssp.core.constants import HTTP_STATUSES
from pynssp.core.auth import Auth

class Apikey(Auth):
    """An Apikey Class Representing an API-key authentication object

    An Apikey object has an API key and a key name.
    An Apikey object can get API data via an API URL.

    :param api_key: A string representing an API key.
    :param key_name: A string representing a header key name for API-key authentication (default is "API-KEY").

    :examples:
        >>> from pynssp import Apikey
        >>>
        >>> myApiProfile = Apikey("my_api_key")
    """

    def __init__(self, api_key=None, key_name="API-KEY"):
        """Initialize a new Apikey object.

        :param api_key: API key string.
        :param key_name: Header key name used for API-key authentication.
        """

        if key_name is None or not isinstance(key_name, str) or key_name.strip() == "":
            raise ValueError("Argument `key_name` must be a non-empty string")
        self.api_key = NSSPContainer(self.__k.encrypt(api_key.encode()))
        self.key_name = key_name

    def get_api_response(self, url):
        """Get API response using API-key header authentication.

        :param url: A string API URL.
        :returns: an object of class response

        """
        if self.api_key is None:
            raise ValueError("Please, set your API key!")

        if not isinstance(url, str) or url.strip() == "":
            raise ValueError("Argument `url` must be a non-empty string")

        headers = {self.key_name: self.__k.decrypt(self.api_key.value).decode()}

        response = get(url, headers=headers)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response
        else:
            raise ValueError(f"Failed to fetch API response: {response.status_code} - {HTTP_STATUSES[str(response.status_code)]}")
