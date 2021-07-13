[![Python package](https://github.com/Blosc/python-caterva/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/Blosc/python-caterva/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/python-caterva/badge/?version=latest)](https://python-caterva.readthedocs.io/en/latest/?badge=latest)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)

# python-caterva

Python wrapper for Caterva.  Still on development.

## Clone repo and submodules

```sh
git clone --recurse-submodules https://github.com/Blosc/python-caterva
```

## Development workflow

### Install requirements

```sh
python -m pip install -r requirements-build.txt
python -m pip install -r requirements.txt
python -m pip install -r requirements-tests.txt
```

### Compile

```sh
python setup.py build_ext --build-type=RelWithDebInfo
```

### Run tests

```sh
PYTHONPATH=. pytest
```

### Installing

```sh
python -m pip install .
```
