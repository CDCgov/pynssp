import pytest
from pynssp import change_dates
from datetime import datetime


def test_change_dates():
    url = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/hospitalSyndromeAlerts?end_date=31Jan2021&start_date=29Jan2021"
    url1 = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/hospitalSyndromeAlerts?end_date=31Jan2021&start_date=15Jan21"
    url2 = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/hospitalSyndromeAlerts?end_date=15Feb21&start_date=29Jan2021"
    url3 = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/hospitalSyndromeAlerts?end_date=15Feb21&start_date=15Jan21"

    assert url1 == change_dates(url, start_date="2021-01-15")
    assert url2 == change_dates(url, end_date="2021-02-15")
    assert url3 == change_dates(url, start_date="2021-01-15", end_date="2021-02-15")
    assert url3 == change_dates(url, start_date=datetime(2021, 1, 15), end_date=datetime(2021, 2, 15))

    with pytest.raises(Exception):
        change_dates(url, start_date="2021-02-15", end_date="2021-01-15")
