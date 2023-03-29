from requests import get
from json import loads
from io import StringIO
from pandas import json_normalize, read_csv
from cryptography.fernet import Fernet
from tempfile import NamedTemporaryFile
from pynssp.core.container import NSSPContainer, APIGraph
from pynssp.core.constants import HTTP_STATUSES


"""
A Token Class Representing a Token object
@decription: A Token object has a token string and a key.
@details: A Token object can get API data via an API URL.
"""
class Token:
    """ 
    Initializes a new Token object.
    @param token: a token string
    @param access_token: type of HTTP authentication. Should be `Bearer` or `Basic`. (Default value = "Bearer")
    """
    def __init__(self, token, access_token = "Bearer"):
        self.__k = Fernet(Fernet.generate_key())
        self.__token = NSSPContainer(self.__k.encrypt(token.encode()))
        self.access_token = access_token
    
    def get_api_response(self, url):
        """
        Get API response
        @param url: a string of API URL
        @return: an object of class response
        """
        headers = {
            'Authorization': "{} {}".
            format(self.access_token, self.__k.decrypt(self.__token.value).decode())
        }
        response = get(url, headers=headers)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response

    def get_api_data(self, url, fromCSV = False, encoding = "utf-8"):
        """
        Get API data
        @param url: a string of API URL
        @param fromCSV: a logical, defines whether data are received in .csv format or .json format (Default value = False)
        @param encoding: an encoding standard (Default value = "utf-8")
        @return: A pandas dataframe
        """
        response_content = self.get_api_response(url).content
        if not fromCSV:
            response_json = loads(response_content)
            return json_normalize(response_json)
        else:
            return read_csv(StringIO(response_content.decode(encoding)))

    def get_api_graph(self, url, file_ext = ".png"):
        """
        Get API graph
        @param url: a string of API URL
        @param file_ext: a non-empty character vector giving the file extension. (Default value = ".png")
        @return: an object of type APIGraph
        """
        response = self.get_api_response(url)
        img_file = NamedTemporaryFile(suffix=file_ext, delete=False)
        img_file.write(response.content)
        return APIGraph(path=img_file, response=response)
