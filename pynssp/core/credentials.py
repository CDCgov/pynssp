import requests
import json
from pandas import json_normalize
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer

class Credentials:
    def __init__(self, username, password):
        self.__k = Fernet(Fernet.generate_key())
        self.username = NSSPContainer(self.__k.encrypt(username.encode()))
        self.password = NSSPContainer(self.__k.encrypt(password.encode()))
    

    def get_api_response(self, url):
        auth = (self.__k.decrypt(self.username.value.encode()), self.__k.decrypt(self.password.value.encode()))
        resp = requests.get(url, auth = auth)
        return resp
    

    def get_api_data(self, url, fromCSV = False):
        if not fromCSV:
            response = self.get_api_response(url)
            response_json = json.loads(response.content)
            return json_normalize(response_json)
        else:
            pass
    

    def get_api_graph(url, file_ext = ".png"):
        pass

