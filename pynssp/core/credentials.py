from requests import get
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer
from pynssp.core.constants import HTTP_STATUSES
from pynssp.core.auth import Auth


class Credentials(Auth):
    """A Token Class Representing a Credentials object

    A Credentials object has a username, a password and a key.
    A Credentials object gets API data via an API URL.

    :param username: a string for username (Default username = None)
    :param password: a string for password (Default password = None)

    :examples:
        >>> from pynssp import Credentials
        >>>
        >>> myProfile = Credentials("user", "pass")
    """

    def __init__(self, username=None, password=None, filename="myProfile"):
        """Initializes a new Credentials object.
        """
        self.__k = Fernet(Fernet.generate_key())
        self.__username = NSSPContainer(self.__k.encrypt(username.encode())) \
            if username is not None else None
        self.__password = NSSPContainer(self.__k.encrypt(password.encode())) \
            if password is not None else None
        self.filename = filename

    def get_api_response(self, url):
        """Get API response

        :param url: a string of API URL
        :returns: an object of class response
        """
        auth = (self.__k.decrypt(self.__username.value).decode(),
                self.__k.decrypt(self.__password.value).decode()) \
            if self.__password is not None else None
        response = get(url, auth=auth) if auth is not None else get(url)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response
