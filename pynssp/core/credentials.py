import requests
import json
import io
from pandas import json_normalize, read_csv
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer
from tempfile import TemporaryFile

class Credentials:
    def __init__(self, username, password):
        self.__k = Fernet(Fernet.generate_key())
        self.username = NSSPContainer(self.__k.encrypt(username.encode()))
        self.password = NSSPContainer(self.__k.encrypt(password.encode()))
    

    def get_api_response(self, url):
        auth = (self.__k.decrypt(self.username.value.encode()), 
                self.__k.decrypt(self.password.value.encode()))
        response = requests.get(url, auth = auth)
        print("%s: %s".format(response.status_code, response.reason))
        if response.status_code == 200:
            return response
    

    def get_api_data(self, url, fromCSV = False, encoding = "utf-8"):
        response_content = self.get_api_response(url).content
        if not fromCSV:
            response_json = json.loads(response_content)
            return json_normalize(response_json)
        else:
            read_csv(io.StringIO(response_content.decode(encoding)))
    

    def get_api_graph(self, url, file_ext = ".png"):
        response = self.get_api_response(url)
        img_file = TemporaryFile(suffix=file_ext).name
        with open(img_file, 'wb') as f:
            f.write(response.content)
        return {'response': response, 'graph': img_file}


