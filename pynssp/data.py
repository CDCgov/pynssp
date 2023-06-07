import pkg_resources
import pandas as pd


def load_simulated_ts():
    """Return a dataframe of simulated time series.

    :examples:
    
        >>> from pynssp.data import load_simulated_ts
        >>> 
        >>> simulated_ts = load_simulated_ts()
        >>> simulated_ts.info()
        ## #   Column  Non-Null Count  Dtype
        ## ---  ------  --------------  -----
        ## 0   date    626 non-null    object
        ## 1   week    626 non-null    int64
        ## 2   year    626 non-null    int64
        ## 3   cases   626 non-null    int64
        ## 4   id      626 non-null    object
        ## dtypes: int64(3), object(2)
        ## memory usage: 24.6+ KB
    """
    
    stream = pkg_resources.resource_stream(__name__, "data/simulated_ts.csv")
    return pd.read_csv(stream)


def get_scenario1():
    """Return a subset of the simulated time series data ('scenario #1').

    :examples:
    
        >>> from pynssp import get_scenario1
        >>> 
        >>> scenario1_ts = get_scenario1()
        >>> scenario1_ts.info()
    """
    simulated_ts = load_simulated_ts()
    return load_simulated_ts()[simulated_ts["id"] == "Scenario #1"]


def get_scenario2():
    """Return a subset of the simulated time series data ('scenario #2').

    :examples:
    
        >>> from pynssp import get_scenario2
        >>> 
        >>> scenario2_ts = get_scenario2()
        >>> scenario2_ts.info()
    """
    simulated_ts = load_simulated_ts()
    return load_simulated_ts()[simulated_ts["id"] == "Scenario #2"]


def load_nssp_stopwords():
    """Return a dataframe of NSSP-curated stopwords.

    :examples:
    
        >>> from pynssp import load_nssp_stopwords
        >>> 
        >>> stopwords = load_nssp_stopwords()
        >>> stopwords.info()
        ## #   Column      Non-Null Count  Dtype
        ## ---  ------      --------------  -----
        ## 1   word        835 non-null    object
        ## 2   type        835 non-null    object
        ## dtypes: int64(1), object(2)
        ## memory usage: 13.2+ KB
    """
    
    stream = pkg_resources.resource_stream(__name__, "data/nssp_stopwords.csv")
    return pd.read_csv(stream)