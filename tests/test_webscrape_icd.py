import pytest
from pynssp.utils import webscrape_icd
import pandas as pd

def test_webscrape_icd_function():
    icd10_2021 = webscrape_icd(icd_version="icd10", year=2021)
    icd10_2020 = webscrape_icd(icd_version="ICD10", year=2020)
    icd9_2014 = webscrape_icd(icd_version="ICD9")

    assert isinstance(icd10_2021, pd.DataFrame)
    assert isinstance(icd10_2020, pd.DataFrame)
    assert isinstance(icd9_2014, pd.DataFrame)

    with pytest.raises(Exception):
        webscrape_icd("icd10")

    with pytest.raises(Exception):
        webscrape_icd("ICD11")

    with pytest.raises(Exception):
        webscrape_icd("ICD10", year=2014)

    with pytest.raises(Exception):
        webscrape_icd("ICD9", year=2020)

    with pytest.raises(Exception):
        webscrape_icd("ICD10", year=2017)

    with pytest.raises(Exception):
        webscrape_icd("ICD10", year=2100)

    with pytest.raises(Exception):
        webscrape_icd("ICD10")