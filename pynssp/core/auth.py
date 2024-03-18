from json import loads
from io import StringIO
from pandas import json_normalize, read_csv
from tempfile import NamedTemporaryFile
from pynssp.core.container import APIGraph


class Auth:
    """An Abstract Auth Class Delineating Methods and Variables Shared by the Token and Credentials Classes
    """

    def __init__(self):
        """Initializes a new Auth object.
        """
        self.filename = None

    def get_api_response(self, url):
        pass

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
        file_name = self.filename + file_ext
        if file is not None:
            file_name = file
        with open(file_name, "wb") as f:
            dump(self, f)
