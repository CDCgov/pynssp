import requests
import json
import io
from pandas import json_normalize, read_csv
from cryptography.fernet import Fernet
from pynssp.core.container import NSSPContainer, APIGraph
from tempfile import TemporaryFile

class Token:
    def __init__(self, token, access_token = "Bearer"):
        self.__k = Fernet(Fernet.generate_key())
        self.token = NSSPContainer(self.__k.encrypt(token.encode()))
        self.access_token = access_token
    
    def get_api_response(self, url):
        headers = {
            'Authorization': "{} {}".
            format(self.access_token, self.__k.decrypt(self.token.value))
            }
        response = requests.get(url, headers=headers)
        print("{}: {}".format(response.status_code, response.reason))
        if response.status_code == 200:
            return response

    def get_api_data(self, url, fromCSV = False, encoding = "utf-8"):
        response_content = self.get_api_response(url).content
        if not fromCSV:
            response_json = json.loads(response_content)
            return json_normalize(response_json)
        else:
            return read_csv(io.StringIO(response_content.decode(encoding)))

    def get_api_graph(self, url, file_ext = ".png"):
        response = self.get_api_response(url)
        img_file = TemporaryFile(suffix=file_ext).name
        with open(img_file, 'wb') as f:
            f.write(response.content)
        return APIGraph(path=img_file, response=response)