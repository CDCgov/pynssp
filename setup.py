#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [ 
    "pandas", "cryptography", "pillow", "scikit-image", "matplotlib", "statsmodels",
    "scipy", "datetime" 
    ]

test_requirements = ['pytest>=3', ]

setup(
    author="Gbedegnon Roseric Azondekon",
    author_email='gazondekon@cdc.gov, roseric_2000@yahoo.fr',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="A Python package for the National Syndromic Surveillance Program (NSSP) and its Community of Practice. A collection of classes and methods to advance the practice of Syndromic Surveillance.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='pynssp',
    name='pynssp',
    packages=find_packages(include=['pynssp', 'pynssp.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cdcgov/pynssp',
    version='0.1.0',
    zip_safe=False,
    package_data={'': ['data/*.csv']},
)
