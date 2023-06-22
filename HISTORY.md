# Release Notes

## 0.1.0 (2023-06-22)
* First release on PyPI.

### Features

#### Credentials Management
* `create_profile()` initializes a `Credentials` object from a username and password.
* `create_token_profile()` initializes a `Token` object from a REST API token.

#### REST API data pulls
* `get_api_response()` retrieves a response from an API service using a provided profile of type `Credentials` or `Token`. Used as a wrapper to the `.get_api_response()` method of a provided profile of type `Credentials` or `Token`.
* `get_api_data()` retrieves data (in JSON or CSV) from an API service using a provided profile of type `Credentials` or `Token`. Used as a wrapper to the `.get_api_data()` method of a provided profile of type `Credentials` or `Token`.
* `get_api_graph()` retrieves a graph from an API service using a provided profile of type `Credentials` or `Token`. Used as a wrapper to the `.get_api_graph()` method of a provided profile of type `Credentials` or `Token`.
* `get_essence_data()` retrieves data from the NSSP-ESSENCE API service using a provided profile of type `Credentials` or `Token`.

#### Anomaly Detection and Trend Classification
* `alert_ewma()` implements the EWMA time series anomaly detection algorithm.
* `alert_regression()` implements the Multiple Adaptive Regression time series anomaly detection algorithm.
* `alert_switch()` implements the Regression/EWMA Switch time series anomaly detection algorithm
* `alert_nbinom()` implements the Negative Binomial Regression time series anomaly detection algorithm.
* `alert_serfling()` implements the original and modified Serfling method for time series anomaly detection.
* `classify_trend()` fits rolling binomial models to a daily time series of percentages or proportions to classify the overall trend.

#### Added Data
* `load_simulated_ts()` loads a dataframe of simulated time series.
* `load_nssp_stopwords()` loads a dataframe of NSSP-curated stopwords.
* `get_scenario1()` loads a a subset of the simulated time series data ("scenario #1").
* `get_scenario2()` loads a a subset of the simulated time series data ("scenario #2").

#### Utility Functions
* `change_dates()` modifies the start and end dates in a given URL to new dates, if provided.
* `webscrape_icd()` scrapes ICD Codes from the Web.

#### Classes
* `Credentials` is an abstract representation of a profile object given a username and password strings.
* `Token` is an abstract representation of a profile object given a token string.
* `APIGraph` is an abstract representation of a graph object returns from an API service.
* `NSSPContainer` encapsulates a value or an object to store.
