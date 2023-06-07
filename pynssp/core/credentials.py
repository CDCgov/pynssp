from requests import get
from json import loads
from io import StringIO
from pandas import json_normalize, read_csv
from cryptography.fernet import Fernet
from tempfile import NamedTemporaryFile
from pynssp.core.container import NSSPContainer, APIGraph
from pynssp.core.constants import HTTP_STATUSES


class Credentials:
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


    def __init__(self, username=None, password=None):
        """Initializes a new Credentials object.
        """
        self.__k = Fernet(Fernet.generate_key())
        self.__username = NSSPContainer(self.__k.encrypt(username.encode())) \
            if username is not None else None
        self.__password = NSSPContainer(self.__k.encrypt(password.encode())) \
            if password is not None else None


    def get_api_response(self, url):
        """Get API response

        :param url: a string of API URL
        :returns: an object of class response

        """
        auth = (self.__k.decrypt(self.__username.value).decode(), 
                self.__k.decrypt(self.__password.value).decode()) \
                if self.__password is not None else None
        response = get(url, auth = auth) if auth is not None else get(url)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response


    def get_api_data(self, url, fromCSV=False, encoding="utf-8", **kwargs):
        """Get API data

        :param url: a string of API URL
        :param fromCSV: a logical, defines whether data are received in .csv format or .json format (Default value = False)
        :param encoding: an encoding standard (Default value = "utf-8")
        :param **kwargs: Additional keyword arguments to pass to `pandas.read_csv()` if `fromCSV` is True.
        :returns: A pandas dataframe
        """
        response_content = self.get_api_response(url).content
        if not fromCSV:
            response_json = loads(response_content)
            return json_normalize(response_json)
        else:
            return read_csv(StringIO(response_content.decode(encoding)), **kwargs)


    def get_api_graph(self, url, file_ext=".png"):
        """Get API graph

        :param url: a string of API URL
        :param file_ext: a non-empty character vector giving the file extension. (Default value = ".png")
        :returns: an object of type APIGraph
        """
        response = self.get_api_response(url)
        img_file = NamedTemporaryFile(suffix=file_ext, delete=False)
        img_file.write(response.content)
        return APIGraph(path=img_file.name, response=response)


    def pickle(self, file=None, file_ext=".pkl"):
        """Save an object of class Credentials to file

        :param file_ext: a non-empty character vector giving the file extension. (Default value = ".pkl")
        :param file:  (Default value = None)
        """
        from pickle import dump
        file_name = "myProfile" + file_ext
        if file is not None:
            file_name = file
        with open(file_name, "wb") as f:
            dump(self, f)
