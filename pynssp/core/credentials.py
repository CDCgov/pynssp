import requests
import json
from pandas import json_normalize

class Credentials:
    def __init__(self, username, password):
        self.username = username
        self.password = password
    

    def get_api_response(url):
        return requests.get(url)
    

    def get_api_data(url, fromCSV = False):
        if not fromCSV:
            response = self.get_api_response(url)
            response_json = json.loads(response.content)
            return json_normalize(response_json)
        else:
            pass
    
    
    def get_api_graph(url, file_ext = ".png"):
        pass

