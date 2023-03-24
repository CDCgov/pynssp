from requests import get
from json import loads
from io import StringIO
from pandas import json_normalize, read_csv
from cryptography.fernet import Fernet
from tempfile import TemporaryFile
from pynssp.core.container import NSSPContainer, APIGraph
from pynssp.core.constants import HTTP_STATUSES

class Credentials:
    def __init__(self, username, password):
        self.__k = Fernet(Fernet.generate_key())
        self.__username = NSSPContainer(self.__k.encrypt(username.encode()))
        self.__password = NSSPContainer(self.__k.encrypt(password.encode()))
    

    def get_api_response(self, url):
        auth = (self.__k.decrypt(self.__username.value), 
                self.__k.decrypt(self.__password.value))
        response = get(url, auth = auth)
        print("{}: {}".format(response.status_code, HTTP_STATUSES[str(response.status_code)]))
        if response.status_code == 200:
            return response
    

    def get_api_data(self, url, fromCSV = False, encoding = "utf-8"):
        response_content = self.get_api_response(url).content
        if not fromCSV:
            response_json = loads(response_content)
            return json_normalize(response_json)
        else:
            return read_csv(StringIO(response_content.decode(encoding)))
    

    def get_api_graph(self, url, file_ext = ".png"):
        response = self.get_api_response(url)
        img_file = TemporaryFile(suffix=file_ext).name
        with open(img_file, 'wb') as f:
            f.write(response.content)
        return APIGraph(path=img_file, response=response)


