# Getting started

## Introduction

The goal of the `pynssp` package is to facilitate access to the Electronic Surveillance System for the Early Notification
of Community-based Epidemics (ESSENCE) via a secure and simplified interface.
In addition, `pynssp` provides methods and functions that streamline data retrieval by abstracting the complexity of Python scripts from users.

In this vignette, we explain how to create an NSSP user profile and provide examples of how to use it to pull data from ESSENCE using the following ESSENCE APIs:

* Time Series data table
* Time Series PNG image
* Table Builder Results
* Data Details (line-level)
* Summary stats
* Alert list detection table
* Time Series data table with stratified, historical alerts (from ESSENCE2)

## Creating an NSSP user profile

We start by loading the `pandas` and the `pynssp` packages.

```python
>>> import pandas as pd
>>> from pynssp.utils import *
```

The next step is to create an NSSP user profile by creating an object of the `Credentials` class.
Here, we use the `create_profile()` function to create a user profile.

```python
>>> myProfile = create_profile()

# Save profile object to file for future use
>>> myProfile.pickle()

# Load profile object
# myProfile = pd.read_pickle("myProfile.pkl")
```

The above code needs to be executed only once. Upon execution, it prompts the user to provide their username and password.

The created `myProfile` object includes the `.get_api_response()`, `.get_api_data()`, and `.get_api_graph()`
methods, each with parameters for pulling ESSENCE data. Alternatively, the `get_api_response()`, `get_api_data()`, and `get_api_graph()`
functions serve as wrappers for their respective methods. The `get_essence_data()` function may be used for NSSP-ESSENCE API URLs.

In the following sections, we show how to pull data from ESSENCE using the seven APIs listed above.

## Time Series Data Table

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/timeSeries?endDate=9Feb2021&medicalGrouping=injury&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&startDate=11Nov2020&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TimeSeries"

# Pull time series data
>>> api_data_ts = get_api_data(url, profile=myProfile) # or api_data_ts = myProfile.get_api_data(url)

>>> api_data_ts.columns

# Extracting embedded dataframe
>>> api_data_ts = pd.json_normalize(api_data_ts["timeSeriesData"][0])

# Preview data
>>> api_data_ts.head()
```

Alternatively, the example below uses the `get_essence_data()` function to achieve the same outcome and extract the embedded dataframe directly when needed.

```python
>>> api_data_ts = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_ts.head()
```

## Time Series Graph from ESSENCE

The example below shows how to retrieve the Time Series Graph from ESSENCE and insert it into a Jupyter notebook.

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/timeSeries/graph?endDate=9Feb2021&medicalGrouping=injury&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&startDate=11Nov2020&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TimeSeries&graphTitle=National%20-%20Injury%20Syndrome%20Daily%20Counts&xAxisLabel=Date&yAxisLabel=Count"

# Data pull from ESSENCE
>>> api_data_graph = get_api_graph(url, profile=myProfile)

# Check the type of api_data_graph
>>> type(api_data_graph)

# Print image file location
>>> print(api_data_graph)

# Insert it into a Jupyter notebook
>>> api_data_graph.plot()
```

From the example above, the variable `api_data_graph` is an `APIGraph` object. In an interactive mode from the Python console, the `.show()` method can be called on the `APIGraph` object to preview the image it contains.

```python
>>> api_data_graph.show()
```

For an exhaustive list of methods that may be called on an `APIGraph` object, please see the `pynssp` documentation.

## Table Builder Results

The CSV option of the Table Builder Results API pulls in data in the tabular format seen on ESSENCE. The JSON option on the other hand, pulls in the data in a long, pivoted format. In the following subsections, we demonstrate how to pull the Table Builder results data with both options.

### CSV option

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/tableBuilder/csv?endDate=31Dec2020&ccddCategory=cdc%20opioid%20overdose%20v3&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=nodetectordetector&startDate=1Oct2020&ageNCHS=11-14&ageNCHS=15-24&ageNCHS=25-34&ageNCHS=35-44&ageNCHS=45-54&ageNCHS=55-64&ageNCHS=65-74&ageNCHS=75-84&ageNCHS=85-1000&ageNCHS=unknown&timeResolution=monthly&hasBeenE=1&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TableBuilder&rowFields=timeResolution&rowFields=geographyhospitaldhhsregion&columnField=ageNCHS"

# Data pull from ESSENCE
>>> api_data_tb_csv = get_api_data(url, fromCSV=True, profile=myProfile)

# Preview data
>>> api_data_tb_csv.head()
```

### JSON option

While the `get_api_data()` function can also be used to pull data from ESSENCE API URLs returning JSON objects, the `get_essence_data()` function has the advantage of not requiring extra JSON parsing, as described in the second section of this tutorial.

```python
>>> url = "https://essence2.syndromicsurveillance.org/nssp_essence/api/tableBuilder?endDate=31Dec2020&ccddCategory=cdc%20opioid%20overdose%20v3&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=nodetectordetector&startDate=1Oct2020&ageNCHS=11-14&ageNCHS=15-24&ageNCHS=25-34&ageNCHS=35-44&ageNCHS=45-54&ageNCHS=55-64&ageNCHS=65-74&ageNCHS=75-84&ageNCHS=85-1000&ageNCHS=unknown&timeResolution=monthly&hasBeenE=1&medicalGroupingSystem=essencesyndromes&userId=2362&aqtTarget=TableBuilder&rowFields=timeResolution&rowFields=geographyhospitaldhhsregion&columnField=ageNCHS"

# Data pull from ESSENCE
>>> api_data_tb_json = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_tb_json.head()
```
## Data Details (line-level)

Similar to the Table Builder Results API, the Data Details (line-level) API provides CSV and JSON data outputs.

### CSV option

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/dataDetails/csv?medicalGrouping=injury&geography=region%20i&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TimeSeries&startDate=31Jan2021&endDate=31Jan2021"

# Data pull from ESSENCE
>>> api_data_dd_csv = get_api_data(url, fromCSV=True, profile=myProfile)

# Preview data
>>> api_data_dd_csv.head()
```

### JSON option

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/dataDetails?endDate=31Jan2021&medicalGrouping=injury&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&startDate=31Jan2021&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=DataDetails"

# Data pull from ESSENCE
>>> api_data_dd_json = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_dd_json.head()
```

## Summary Stats

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/summaryData?endDate=31Jan2021&medicalGrouping=injury&geography=region%20i&percentParam=noPercent&geographySystem=hospitaldhhsregion&datasource=va_hosp&detector=probrepswitch&startDate=29Jan2021&timeResolution=daily&medicalGroupingSystem=essencesyndromes&userId=455&aqtTarget=TimeSeries"

# Data pull from ESSENCE
>>> api_data_ss = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_ss.head()
```

## Alert List Detection Table

Since the Alert List API provides programmatic access to the Alert List table in the ESSENCE user interface by patient region or hospital region, we provide two data pull use cases in the following subsections:

### Alert List Detection Table by Patient Region

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/regionSyndromeAlerts?end_date=31Jan2021&start_date=29Jan2021"

# Data pull from ESSENCE
>>> api_data_alr = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_alr.head()
```

### Alert List Detection Table by Hospital Region

```python
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/hospitalSyndromeAlerts?end_date=31Jan2021&start_date=29Jan2021"

# Data pull from ESSENCE
>>> api_data_alh = get_api_data(url, profile=myProfile)

# Preview data
>>> api_data_alh.head()
```

## Time Series data table with stratified, historical alerts (from ESSENCE2)

As of February 10, 2023, this functionality is available from ESSENCE2. Therefore, if your ESSENCE 2 credentials are different from the ones defined above for ESSENCE, you will need to create another profile object for ESSENCE 2 and use it to run the code below. In this example, we assume the same user profile is used for both ESSENCE and ESSENCE 2.

```python
>>> url = "https://essence2.syndromicsurveillance.org/nssp_essence/api/timeSeries?endDate=9Feb2021&ccddCategory=cdc%20pneumonia%20ccdd%20v1&ccddCategory=cdc%20coronavirus-dd%20v1&ccddCategory=cli%20cc%20with%20cli%20dd%20and%20coronavirus%20dd%20v2&percentParam=ccddCategory&geographySystem=hospitaldhhsregion&datasource=va_hospdreg&detector=probrepswitch&startDate=11Nov2020&timeResolution=daily&hasBeenE=1&medicalGroupingSystem=essencesyndromes&userId=2362&aqtTarget=TimeSeries&stratVal=ccddCategory&multiStratVal=geography&graphOnly=true&numSeries=3&graphOptions=multipleSmall&seriesPerYear=false&nonZeroComposite=false&removeZeroSeries=true&startMonth=January&stratVal=ccddCategory&multiStratVal=geography&graphOnly=true&numSeries=3&graphOptions=multipleSmall&seriesPerYear=false&startMonth=January&nonZeroComposite=false"

# Data pull from ESSENCE
>>> api_data_tssh = get_essence_data(url, profile=myProfile)

# Preview data
>>> api_data_tssh.head()
```
