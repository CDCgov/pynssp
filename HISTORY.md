# Release Notes

## 0.1.1 (2026-08-19)

### Bug fixes
* The install issue related to package data has been fixed.
* The package info to document Python support up to 3.14 has been updated
* The `change_dates()` function has been upgraded to fix flip-flopping start and end date values.
* The EWMA detector has been updated to closely match the `Rnssp`'s implementation. In addition, the `alert_ewma()` function has been updated with a patch to prevent errors when detector is applied to a zero series.

### New Features

#### Anomaly Detection
* The Farrington Temporal Detector algorithm `alert_farrington()` has been added for weekly time series of counts spanning multiple years.

#### Credentials Management
* A new abstract class `Auth` has been added.
* Both the `Credentials` and `Token` classes inherit the `Auth` class.
* A new `Apikey` class has been added for API services token use.

#### Classes
* The `Auth` class delineates methods Shared by the `Token`, `Credentials`, and `Apikey` Classes.
The new `Apikey` class inherits the `Auth` class, and has the same methods as the `Credentials` and `Token` classes.

#### Utility Functions
* A new `create_apikey_profile()` function has been added as a wrapper to the `Apikey` class.


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
