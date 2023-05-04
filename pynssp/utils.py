import re
import pandas as pd
from .core.credentials import Credentials
from .core.token import Token
from datetime import datetime, date
from dateutil.parser import parse as date_parser
from getpass import getpass

def change_dates(url, start_date=None, end_date=None):
    """Changes the start and end dates in a given URL to new dates, if provided.

    :param url: str): The URL containing the start and end dates to be changed.
    :param start_date: str): A new start date to replace the existing start date in the URL. (Default value = None)
    :param end_date: str): A new end date to replace the existing end date in the URL. (Default value = None)
    :returns: The modified URL with the new start and end dates.
    :examples:
    
        >>> from pynssp.utils import change_dates
        >>> url = "https://example.com/data?startDate=01Jan2022&endDate=31Dec2022"
        >>> change_dates(url, start_date="01Jan2021", end_date="31Dec2021")
    """
    
    # Assert that the input URL is a string.
    assert isinstance(url, str), "URL must be a string."
    
    # Define regular expression patterns to match prefixes of the start and end date parameters.
    epref = re.search(r"(endDate|end_date)=\d+[A-Za-z]+\d+", url).group(1) + "="
    spref = re.search(r"(startDate|start_date)=\d+[A-Za-z]+\d+", url).group(1) + "="
    
    # Extract the current start and end dates from the URL string.
    old_end = re.search(r"(endDate|end_date)=\d+[A-Za-z]+\d+", url).group(0).replace(epref, "")
    old_start = re.search(r"(startDate|start_date)=\d+[A-Za-z]+\d+", url).group(0).replace(spref, "")
    
    # If new start or end dates are provided, replace the old dates with the new dates.
    new_end = old_end
    new_start = old_start
    if end_date is not None:
        if isinstance(end_date, date):
            new_end = end_date.strftime("%d%b%y").strip()
        else:
            new_end = date_parser(end_date).strftime("%d%b%y").strip()
    if start_date is not None:
        if isinstance(start_date, date):
            new_start = start_date.strftime("%d%b%y").strip()
        else:
            new_start = date_parser(start_date).strftime("%d%b%y").strip()
    
    # Convert new start and end dates to datetime objects for comparison.
    new_startd = datetime.strptime(new_start, "%d%b%Y") if len(new_start) > 7 else datetime.strptime(new_start, "%d%b%y")
    new_endd = datetime.strptime(new_end, "%d%b%Y") if len(new_end) > 7 else datetime.strptime(new_end, "%d%b%y")
    
    # Check that the new start date is not after the new end date.
    if new_startd > new_endd:
        raise ValueError(f"Start Date '{new_start}' is posterior to End Date '{new_end}'.")
    
    # Replace the old start and end dates with the new start and end dates in the URL string.
    new_url = url.replace(old_end, new_end).replace(old_start, new_start)
    
    # Remove any whitespace characters from the modified URL string.
    new_url = re.sub(r"\s+", "", new_url)
    
    return new_url


def create_profile(username=None, password=None):
    """Create a new user profile with the given username and password.

    :param username: A string representing the username. If not provided, the user will be prompted to enter it.
    :param password: A string representing the user's password. If not provided, the user will be prompted to enter it securely.
    :return: A new Credentials object with the given username and password.
    :examples:
    
        >>> from pynssp.utils import create_profile
        >>> myProfile = create_profile()
    """
    if username is None:
        username = input("Please enter your username: ")
    if password is None:
        password = getpass()
    return Credentials(username=username, password=password)


def create_token_profile(token=None, access_token="Bearer"):
    """Create a new token profile with the given token and authentication type.

    :param token: A string representing the token. If not provided, the user will be prompted to enter it securely.
    :param auth_type: A string representing the authentication type. Defaults to "Bearer".
    :return: A new Token object with the given token and authentication type.
    :examples:
    
        >>> from pynssp.utils import create_token_profile
        >>> myTokenProfile = create_token_profile()
    """
    if token is None:
        token = getpass(prompt="Enter/Paste a token: ")
    return Token(token=token, access_token=access_token)


def get_api_response(url, profile=None):
    """Retrieve a response from an API using the provided profile.

    :param url: A string representing the URL of the API endpoint.
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` or `pynssp.core.token.Token`.
    :return: The response object returned by the API.
    :examples:
    
        >>> from pynssp.utils import *
        >>> myProfile = create_profile()
        >>> url = "http://httpbin.org/json"
        >>> response = get_api_response(url, profile=myProfile)
    """
    try:
        return profile.get_api_response(url=url)
    except (NameError, AttributeError):
        raise ValueError("Invalid profile name or missing `get_api_response` method.")


def get_api_data(url, fromCSV=False, profile=None, encoding="utf-8", **kwargs):
    """Retrieve data from an API using the provided profile.

    :param url: A string representing the URL of the API endpoint.
    :param fromCSV: A boolean indicating whether the data should be retrieved from a CSV file. Defaults to False.
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` or `pynssp.core.token.Token`.
    :param kwargs: Additional keyword arguments to be passed to the profile's get_api_data method.
    :return: The data retrieved from the API.
    :examples:
    
        >>> from pynssp.utils import *
        >>> myProfile = create_profile()
        >>> url = "http://httpbin.org/json"
        >>> api_data = get_api_data(url, profile=myProfile)
    """
    try:
        return profile.get_api_data(url=url, fromCSV=fromCSV, encoding=encoding, **kwargs)
    except (NameError, AttributeError):
        raise ValueError("Invalid profile name or missing `get_api_data` method.")


def get_api_graph(url, file_ext=".png", profile=None):
    """Retrieve a graph from an API using the provided profile.

    :param url: A string representing the URL of the API endpoint.
    :param file_ext: A string representing the file extension of the graph. Defaults to ".png".
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` or `pynssp.core.token.Token`.
    :return: The graph retrieved from the API.
    :examples:
    
        >>> from pynssp.utils import *
        >>> myProfile = create_profile()
        >>> url = "http://httpbin.org/image/png"
        >>> api_graph = get_api_response(url, profile=myProfile)
    """
    try:
        return profile.get_api_graph(url=url, file_ext=file_ext)
    except (NameError, AttributeError):
        raise ValueError("Invalid profile name or missing `get_api_graph` method.")


def get_essence_data(url, start_date=None, end_date=None, profile=None, **kwargs):
    """Retrieve data from the NSSP-ESSENCE API using the provided profile.

    :param url: A string representing the URL of the NSSP-ESSENCE API endpoint.
    :param start_date: A string representing the start date of the data to retrieve.
    :param end_date: A string representing the end date of the data to retrieve.
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` or `pynssp.core.token.Token`.
    :param kwargs: Additional arguments to be passed to the get_api_data function.
    :return: The data retrieved from the NSSP-ESSENCE API.
    :examples:
    
        >>> from pynssp.utils import *
        >>> myProfile = create_profile()
        >>> url = "https://essence2.syndromicsurveillance.org/nssp_essence/api/timeSeries/graph?endDate=25Jun2022&geography=&percentParam=noPercent&datasource=va_hosp&startDate=25Jun2021&medicalGroupingSystem=essencesyndromes&userId=3751&aqtTarget=TimeSeries&ccddCategory=&geographySystem=hospitalregion&detector=probrepswitch&timeResolution=daily"
        >>> api_data = get_essence_data(url, profile=myProfile)
    """
    if profile is None:
        raise ValueError("Please, provide a profile object of type `Credentials` or `Token`!")

    api_type = re.search(r"(?<=api/).+(?=\?)", url).group()

    if not api_type:
        raise ValueError("URL is not of NSSP-ESSENCE type. Check your URL or use get_api_data instead!")

    url_new = change_dates(url, start_date, end_date)
    
    if api_type == "timeSeries":
        api_data = profile.get_api_data(url_new, **kwargs)
        return pd.json_normalize(api_data["timeSeriesData"][0])
    elif api_type == "timeSeries/graph":
        return profile.get_api_graph(url_new)
    elif api_type == "tableBuilder/csv":
        return profile.get_api_data(url_new, fromCSV=True, **kwargs)
    elif api_type == "tableBuilder":
        return profile.get_api_data(url_new, **kwargs)
    elif api_type == "dataDetails":
        api_data = profile.get_api_data(url_new, **kwargs)
        return pd.json_normalize(api_data["dataDetails"][0])
    elif api_type == "dataDetails/csv":
        return profile.get_api_data(url_new, fromCSV=True, **kwargs)
    elif api_type == "summaryData":
        api_data = profile.get_api_data(url_new, **kwargs)
        return pd.json_normalize(api_data["summaryData"][0])
    elif api_type == "alerts/regionSyndromeAlerts":
        api_data = profile.get_api_data(url_new, **kwargs)
        return pd.json_normalize(api_data["regionSyndromeAlerts"][0])
    elif api_type == "alerts/hospitalSyndromeAlerts":
        api_data = profile.get_api_data(url_new, **kwargs)
        return pd.json_normalize(api_data["hospitalSyndromeAlerts"][0])
    else:
        raise ValueError("URL is not of ESSENCE type. Check your URL or use the `get_api_data()` function instead!")
