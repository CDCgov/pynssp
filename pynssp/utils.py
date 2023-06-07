import re
import pandas as pd
import requests
import tempfile
import os
import zipfile
from .core.credentials import Credentials
from .core.token import Token
from datetime import datetime, date
from dateutil.parser import parse as date_parser
from getpass import getpass

def change_dates(url, start_date=None, end_date=None):
    """Changes the start and end dates in a given URL to new dates, if provided.

    :param url: str): The URL containing the start and end dates to be changed.
    :param start_date: str): A new start date to replace the existing 
        start date in the URL. (Default value = None)
    :param end_date: str): A new end date to replace the existing end date in the URL. 
        (Default value = None)
    :returns: The modified URL with the new start and end dates.
    :examples:
    
        >>> from pynssp import change_dates
        >>> 
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
    new_startd = datetime.strptime(new_start, "%d%b%Y") \
        if len(new_start) > 7 \
        else datetime.strptime(new_start, "%d%b%y")
    
    new_endd = datetime.strptime(new_end, "%d%b%Y") \
        if len(new_end) > 7 \
        else datetime.strptime(new_end, "%d%b%y")
    
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

    :param username: A string representing the username. If not provided, 
        the user will be prompted to enter it.
    :param password: A string representing the user's password. 
        If not provided, the user will be prompted to enter it securely.
    :return: A new Credentials object with the given username and password.
    :examples:
    
        >>> from pynssp import create_profile
        >>> 
        >>> myProfile = create_profile()
    """
    if username is None:
        username = input("Please enter your username: ")
    if password is None:
        password = getpass()
    return Credentials(username=username, password=password)


def create_token_profile(token=None, access_token="Bearer"):
    """Create a new token profile with the given token and authentication type.

    :param token: A string representing the token. If not provided, 
        the user will be prompted to enter it securely.
    :param auth_type: A string representing the authentication type. 
        Defaults to "Bearer".
    :return: A new Token object with the given token and authentication type.
    :examples:
    
        >>> from pynssp import create_token_profile
        >>> 
        >>> myTokenProfile = create_token_profile()
    """
    if token is None:
        token = getpass(prompt="Enter/Paste a token: ")
    return Token(token=token, access_token=access_token)


def get_api_response(url, profile=None):
    """Retrieve a response from an API using the provided profile.

    :param url: A string representing the URL of the API endpoint.
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` 
        or `pynssp.core.token.Token`.
    :return: The response object returned by the API.
    :examples:
    
        >>> from pynssp import *
        >>> 
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
    :param fromCSV: A boolean indicating whether the data should be 
        retrieved from a CSV file. Defaults to False.
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` 
        or `pynssp.core.token.Token`.
    :param kwargs: Additional keyword arguments to be passed to 
        the profile's get_api_data method.
    :return: The data retrieved from the API.
    :examples:
    
        >>> from pynssp import *
        >>> 
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
    :param file_ext: A string representing the file extension of 
        the graph. Defaults to ".png".
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` 
        or `pynssp.core.token.Token`.
    :return: The graph retrieved from the API.
    :examples:
    
        >>> from pynssp import *
        >>> 
        >>> myProfile = create_profile()
        >>> url = "http://httpbin.org/image/png"
        >>> api_graph = get_api_graph(url, profile=myProfile)
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
    :param profile: An profile object of class `pynssp.core.credentials.Credentials` 
        or `pynssp.core.token.Token`.
    :param kwargs: Additional arguments to be passed to the get_api_data function.
    :return: The data retrieved from the NSSP-ESSENCE API.
    :examples:
    
        >>> from pynssp import *
        >>> 
        >>> myProfile = create_profile()
        >>> url = "https://essence2.syndromicsurveillance.org/nssp_essence/api/timeSeries/graph?endDate=25Jun2022&geography=&percentParam=noPercent&datasource=va_hosp&startDate=25Jun2021&medicalGroupingSystem=essencesyndromes&userId=3751&aqtTarget=TimeSeries&ccddCategory=&geographySystem=hospitalregion&detector=probrepswitch&timeResolution=daily"
        >>> api_data = get_essence_data(url, profile=myProfile)
        >>> api_data.info()
        
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


def webscrape_icd(icd_version="ICD10", year=None):
    """ICD Code Web Scraper

    Function to web scrape ICD discharge diagnosis code sets from the CDC FTP server (for ICD-10)
    or CMS website (for ICD-9). If pulling ICD-10 codes, by default the function will search for 
    the most recent year's code set publication by NCHS. Users can specify earlier publication 
    years back to 2019 if needed. The ICD-9 option will only web scrape the most recent, 
    final ICD-9 code set publication (2014) from the CMS website. 
    This function will return an error message if the FTP server or CMS website is 
    unresponsive or if a timeout of 60 seconds is reached. 
    The result is a dataframe with 3 fields: code, description, and set 
    (ICD version concatenated with year). 
    Codes are standardized to upper case with punctuation and extra 
    leading/tailing white space removed to enable successful joining.

    :param icd_version: The version of ICD codes to retrieve. Default is "ICD10".
    :param year: The year for which to retrieve the ICD codes. 
        If not provided, the current year will be used. (Default value = None)
    :returns: A DataFrame containing the ICD codes and descriptions.
    :examples:

        >>> # Example 1
        >>> from pynssp import webscrape_icd
        >>> 
        >>> icd9 = webscrape_icd(icd_version = "ICD9")
        >>> icd9.head()

        >>> # Example 2
        >>> from pynssp import webscrape_icd
        >>> 
        >>> icd10_2021 = webscrape_icd(icd_version="ICD10", year=2021)
        >>> icd10_2021.info()

        >>> # Example 3
        >>> from pynssp import webscrape_icd
        >>> 
        >>> icd10_2020 = webscrape_icd(icd_version="ICD10", year=2020)
        >>> icd10_2020.info()
    """
    
    icd_version = icd_version.upper()
    
    if not re.search("ICD10|ICD9", icd_version):
        raise ValueError("ICD version argument icd_version must be 'ICD9' or 'ICD10'")
        
    if icd_version == "ICD9" and year is not None:
        raise ValueError("Argument year only applies for ICD10")
        
    if icd_version == "ICD10" and year is not None:
        if year <= 2018:
            raise ValueError("ICD-10 code sets prior to 2019 are not supported")
            
        if year > datetime.now().year + 1:
            raise ValueError("Argument year cannot be greater than the upcoming year.")
    
    if icd_version == "ICD10":
        ftp_url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/"

        root_folders = requests.get(ftp_url).text.split("\n")[2]

        current_year = None
        if year:
            current_year = year == int(pd.Timestamp.today().strftime("%Y"))
        else:
            current_year = True

        if year is None or current_year:
            year = int(pd.Timestamp.today().strftime("%Y"))
            path = re.findall("(?<=HREF=\").*?(?=\")", root_folders)
            years = []
            for path in path:
                match = re.search(r"\d{4}", path)
                if match and (match.group(0) not in years):
                    years.append(match.group(0))
                    
            
            if str(year) in years:
                path = f"pub/Health_Statistics/NCHS/Publications/ICD10CM/{year}/"
            else:
                raise ValueError(f"No ICD10 found for the year {year}.")

            res = requests.get(f"https://ftp.cdc.gov/{path}")
            path_files = res.text.split("\n")[2]

            pattern = r'(?<=HREF=")[^"]+\.\w+(?=">)'
            file_list = re.findall(pattern, path_files)

            pattern2 = "(code_descriptions|code%20descriptions|icd10cm_codes_\\d{4})"
            file_match = [re.search(pattern2, re.sub("[ -]", "_", f.lower())) is not None for f in file_list]

            if all(not x for x in file_match):
                raise ValueError(f"The {pd.Timestamp.today().year} code description file is not yet available. Please try a previous year.")
            else:
                file = f"https://ftp.cdc.gov/{file_list[file_match.index(True)]}"
                file_ext = os.path.splitext(file)[1].split()

                file_idx = [i for i, ext in enumerate(file_ext) if ext == ".zip"]

                if len(file_idx) == 0:
                    raise ValueError("No ZIP file found in directory")

                temp_file = tempfile.NamedTemporaryFile(suffix=file_ext[0], delete=False)

                with temp_file:
                    temp_dir = os.path.dirname(temp_file.name)
                    response = requests.get(file, stream=True)
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=1024):
                        temp_file.write(chunk)

                with zipfile.ZipFile(temp_file.name, "r") as zip_ref:
                    file_list = zip_ref.namelist()
                    
                    for f in file_list:
                        if re.match(f"Code Descriptions/icd10cm-codes-{year}.txt", f):
                            file_name = f
                            break
                    else:
                        raise ValueError(f"No file matching code description file found for {year}")

                    zip_ref.extract(file_name, path=temp_dir)
                    file_path = os.path.join(temp_dir, file_name)

                    icd_dictionary = pd.read_csv(file_path, sep="\t", header=None, names=["code_combo"]) \
                        .assign(code_combo=lambda df: df["code_combo"].str.replace("\\s{3,5}", "_", regex=True)) \
                        .assign(code=lambda df: df["code_combo"].str.extract("^(.{1,5})")) \
                        .assign(code=lambda df: df["code"].str.replace("_", "", regex=True)) \
                        .assign(description=lambda df: df["code_combo"].str.extract("^(?:.{1,5})(.*)$")) \
                        .assign(description=lambda df: df["description"].str.replace("_", "", regex=True)) \
                        .assign(set=f"ICD-10 {year}")
                    icd_dictionary = icd_dictionary[["code", "description", "set"]]
            return icd_dictionary
        else:
            year
            path = f"pub/Health_Statistics/NCHS/Publications/ICD10CM/{year}/"
            file_path = f"https://ftp.cdc.gov/{path}icd10cm_codes_{year}.txt"

            icd_dictionary = pd.read_csv(file_path, sep="\t", header=None, names=["code_combo"]) \
                .assign(code_combo=lambda df: df["code_combo"].str.replace("\\s{3,5}", "_", regex=True)) \
                .assign(code=lambda df: df["code_combo"].str.extract("^(.{1,5})")) \
                .assign(code=lambda df: df["code"].str.replace("_", "", regex=True)) \
                .assign(description=lambda df: df["code_combo"].str.extract("^(?:.{1,5})(.*)$")) \
                .assign(description=lambda df: df["description"].str.replace("_", "", regex=True)) \
                .assign(set=f"ICD-10 {year}")
            icd_dictionary = icd_dictionary[["code", "description", "set"]]

            return icd_dictionary
    else:
        base_url = "https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes"
        icd_file = "Downloads/ICD-9-CM-v32-master-descriptions.zip"
        cms_url = f"{base_url}/{icd_file}"

        temp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        temp_dir = os.path.dirname(temp_file.name)
        
        with temp_file:
            temp_dir = os.path.dirname(temp_file.name)
            try:
                response = requests.get(cms_url, stream=True)
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024):
                    temp_file.write(chunk)
            except:
                raise Exception("Error in webscrape_icd: ICD-9 webscrape failed. CMS website is currently unresponsive.")
            
        with zipfile.ZipFile(temp_file.name, "r") as zip_file:
            file_name = None
            for info in zip_file.infolist():
                if info.filename == "CMS32_DESC_LONG_DX.txt":
                    file_name = info.filename
                    zip_file.extract(info.filename, path=temp_dir)

        file_path = f"{temp_dir}/{file_name}"
        file_year = 2014

        icd_dictionary = pd.read_csv(file_path, sep="\t", header=None, names=["code_combo"], encoding = "ISO-8859-1") \
            .assign(code_combo=lambda df: df["code_combo"].str.replace("\\s{3,5}", "_", regex=True)) \
            .assign(code=lambda df: df["code_combo"].str.extract("^(.{1,5})")) \
            .assign(code=lambda df: df["code"].str.replace("_", "", regex=True)) \
            .assign(description=lambda df: df["code_combo"].str.extract("^(?:.{1,5})(.*)$")) \
            .assign(description=lambda df: df["description"].str.replace("_", "", regex=True)) \
            .assign(set=f"ICD-9 {file_year}")
        icd_dictionary = icd_dictionary[["code", "description", "set"]]

        return icd_dictionary