# About pynssp 
[![pynssp Pipeline](https://github.com/cdcent/pynssp/actions/workflows/check-standard.yaml/badge.svg)](https://github.com/cdcent/pynssp/actions/workflows/check-standard.yaml)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://github.com/CDCgov/pynssp)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
## Overview
`pynssp` is a Python package for the National Syndromic Surveillance Program (NSSP) and its Community of Practice. A collection of classes and methods to advance the practice of Syndromic Surveillance. It serves as a Python alternative to the [Rnssp R package](https://cdcgov.github.io/Rnssp).

## Installation

### Stable release

To install `pynssp`, run this command in your terminal:

```console
$ pip install pynssp
```

This is the preferred method to install `pynssp`, as it will always install the most recent stable release.

### Development version

You can install the development version of `pynssp` from Github:

```console
$ pip install git+https://github.com/CDCgov/pynssp.git
```

## Usage
```python
>>> from pynssp import *
>>> from datetime import date, timedelta
>>> import pandas as pd


## Creating a user profile (token)
>>> myProfile = create_token_profile()

## Creating a user profile (username and password)
>>> myProfile = create_profile()

## JSON URL from NSSP-ESSENCE API
>>> url = "https://essence.syndromicsurveillance.org/nssp_essence/api/alerts/regionSyndromeAlerts?end_date=31Jan2021&start_date=29Jan2021"

## Update Start and End dates in NSSP-ESSENCE API URL
>>> startDate = date.today() - timedelta(days=30)
>>> endDate = date.today()

>>> url = change_dates(url, start_date = startDate, end_date = endDate)

## Pull Time Series Data from NSSP-ESSENCE
>>> api_data = get_api_data(url, profile=myProfile)

## Inspect data object structure
>>> api_data.columns

## Extract table of interest
>>> api_data = pd.json_normalize(api_data["regionSyndromeAlerts"][0])

## Get a glimpse of the pulled dataset
>>> api_data.head()
```

## Contributing to this project
Should you want to contribute to this project, submit a push request to this Github repository and consider submitting a request to be added as a developer to gazondekon@cdc.gov.

## Getting Help
If you encounter a clear bug, please consider emailing the author at gazondekon@cdc.gov and/or [file an issue](https://github.com/CDCgov/pynssp/issues) with a minimal reproducible example.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC ? 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
