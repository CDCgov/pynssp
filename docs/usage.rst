=====
Get started
=====

Introduction
=====

The goal of the `pynssp` package is to facilitate the access to the Electronic Surveillance System for the Early Notification 
of Community-based Epidemics (ESSENCE) via a secure and simplified interface. 
In addition, `pynssp` provides methods and functions that streamline the data pull by abstracting the complexity of the R codes from the users.

In this vignette, we explained how to create an NSSP user profile, and provide various examples to how to use it to pull data from ESSENCE using the following ESSENCE APIs:

* Time series data table
* Time series png image
* Table builder results
* Data details (line level)
* Summary stats
* Alert list detection table
* Time series data table with stratified, historical alerts (from ESSENCE2)

Creating an NSSP user profile
=====

We start by loading the `pynssp` package::

    # Loading useful R packages...
    import pandas as pd
    from pynssp.utils import *

The next step is to create an NSSP user profile by creating an object of the class `Credentials`. 
Here, we use the `create_profile()` function to create a user profile::
    
    # Creating an ESSENCE user profile
    myProfile = create_profile()
    
    # Save profile object to file for future use
    myProfile.pickle()

The above code needs to be executed only once. Upon execution, it prompts the user to provide his username and password.

The created myProfile object comes with the `.get_api_response()`, `.get_api_data()`, `.get_api_graph()` and `.get_api_tsgraph()` 
methods with various parameters to pull ESSENCE data. Alternatively, the `get_api_response()`, `get_api_data()`, `get_api_graph()` 
functions serve as wrappers to their respective methods. The `get_essence_data()` function may be used for NSSP-ESSENCE API URLs.

In the following sections, we show how to pull data from ESSENCE using the seven APIs listed above.

Time Series Data Table
=====
::

    # URL from ESSENCE JSON Time Series Data Table API
    url = "https://essence.syndromicsurveillance.org/nssp_essence/api/timeSeries?endDate=9Feb2021&medicalGrouping=injury&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&startDate=11Nov2020&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TimeSeries"
    
    # Pull time series data
    api_data_ts = get_api_data(url, profile=myProfile) # or api_data_ts = myProfile.get_api_data(url)
    
    api_data_ts.columns
    
    # Extracting embedded dataframe
    api_data_ts = pd.json_normalize(api_data_ts["timeSeriesData"][0])
    
    # Preview data
    api_data_ts.head()

Alternatively, the example below with the `get_essence_data()` function achieves the same outcome directly extracting the embedded dataframe when needed::

    # Pull time series data
    api_data_ts = get_essence_data(url, profile=myProfile)
    
    # Preview data
    api_data_ts.head()
