from requests import get
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer, APIGraph
from pynssp.core.constants import HTTP_STATUSES
from pynssp.core.auth import Auth


class Token(Auth):
    """A Token Class Representing a Token object

    A Token object has a token string and a key.
    A Token object can get API data via an API URL.

    :param token: a token string
    :param access_token: type of HTTP authentication.
        Should be "Bearer" or "Basic". (Default value = "Bearer")
    :ivar access_token: HTTP authentication type.

    :examples:
        >>> from pynssp import Token
        >>>
        >>> myTokenProfile = Token("abc123")
    """

    def __init__(self, token, access_token="Bearer", filename="tokenProfile"):
        """Initializes a new Token object.
        """
        self.__k = Fernet(Fernet.generate_key())
        self.__token = NSSPContainer(self.__k.encrypt(token.encode()))
        self.access_token = access_token
        self.filename = filename

    def get_api_response(self, url):
        headers = {
            "Authorization": "{} {}".
            format(self.access_token, self.__k.decrypt(self.__token.value).decode())
        }
        response = get(url, headers=headers)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response
