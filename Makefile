.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8
.DEFAULT_GOAL := help

PYTHON ?= python

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

define CLEAN_PYSCRIPT
import pathlib
import shutil
import sys

root = pathlib.Path(".")
mode = sys.argv[1]


def remove_path(path):
	p = pathlib.Path(path)
	if p.is_dir():
		shutil.rmtree(p, ignore_errors=True)
	else:
		try:
			p.unlink()
		except FileNotFoundError:
			pass


def remove_glob(pattern):
	for p in root.rglob(pattern):
		if p.is_dir():
			shutil.rmtree(p, ignore_errors=True)
		else:
			try:
				p.unlink()
			except FileNotFoundError:
				pass


if mode == "build":
	for item in ("build", "dist", ".eggs"):
		remove_path(item)
	remove_glob("*.egg-info")
	remove_glob("*.egg")
elif mode == "pyc":
	for pattern in ("*.pyc", "*.pyo", "*~", "__pycache__"):
		remove_glob(pattern)
elif mode == "test":
	for item in (".tox", ".coverage", "htmlcov", ".pytest_cache"):
		remove_path(item)
else:
	raise ValueError(f"Unknown clean mode: {mode}")
endef
export CLEAN_PYSCRIPT

define REMOVE_PATHS_PYSCRIPT
import pathlib
import shutil
import sys

for raw in sys.argv[1:]:
	p = pathlib.Path(raw)
	if p.is_dir():
		shutil.rmtree(p, ignore_errors=True)
	else:
		try:
			p.unlink()
		except FileNotFoundError:
			pass
endef
export REMOVE_PATHS_PYSCRIPT

define COPY_PATH_PYSCRIPT
import pathlib
import shutil
import sys

src = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
if src.is_dir():
	if dest.exists():
		shutil.rmtree(dest, ignore_errors=True)
	shutil.copytree(src, dest)
else:
	if dest.is_dir():
		shutil.copy2(src, dest / src.name)
	else:
		dest.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(src, dest)
endef
export COPY_PATH_PYSCRIPT

define LIST_DIST_PYSCRIPT
import pathlib

dist = pathlib.Path("dist")
if not dist.exists():
	print("dist/ does not exist")
else:
	for p in sorted(dist.iterdir()):
		size = p.stat().st_size
		print(f"{p} ({size} bytes)")
endef
export LIST_DIST_PYSCRIPT

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	$(PYTHON) -c "$$CLEAN_PYSCRIPT" build

clean-pyc: ## remove Python file artifacts
	$(PYTHON) -c "$$CLEAN_PYSCRIPT" pyc

clean-test: ## remove test and coverage artifacts
	$(PYTHON) -c "$$CLEAN_PYSCRIPT" test

lint/flake8: ## check style with flake8
	flake8 pynssp tests

lint: lint/flake8 ## check style

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	$(PYTHON) -m coverage run --source pynssp -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	$(PYTHON) -c "$$REMOVE_PATHS_PYSCRIPT" docs/pynssp.rst docs/modules.rst
	sphinx-apidoc -o docs/ pynssp
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

pdfdocs: ## generate Sphinx PDF documentation, including API docs
	$(MAKE) -C docs latexpdf -i
	$(PYTHON) -c "$$COPY_PATH_PYSCRIPT" docs/_build/latex/pynssp.pdf docs/pynssp.pdf

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	$(PYTHON) -c "$$LIST_DIST_PYSCRIPT"

install: clean ## install the package to the active Python's site-packages
	$(PYTHON) setup.py install
