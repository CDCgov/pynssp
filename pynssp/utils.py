import re
from datetime import datetime

def change_dates(url, start_date=None, end_date=None):
    """Changes the start and end dates in a given URL to new dates, if provided.

    :param url: str): The URL containing the start and end dates to be changed.
    :param start_date: str): A new start date to replace the existing start date in the URL. (Default value = None)
    :param end_date: str): A new end date to replace the existing end date in the URL. (Default value = None)
    :returns: The modified URL with the new start and end dates.
    :examples
        from pynssp.utils import *
        url = "https://example.com/data?startDate=01Jan2022&endDate=31Dec2022"
        change_dates(url, start_date="01Jan2021", end_date="31Dec2021")
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
        new_end = datetime.strptime(end_date, "%d%b%Y").strftime("%d%b%Y").strip()
    if start_date is not None:
        new_start = datetime.strptime(start_date, "%d%b%Y").strftime("%d%b%Y").strip()
    
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
