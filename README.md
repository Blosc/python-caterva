[![Build Status](https://dev.azure.com/blosc/caterva/_apis/build/status/Blosc.cat4py?branchName=master)](https://dev.azure.com/blosc/caterva/_build/latest?definitionId=1&branchName=master)
![Coverage](https://img.shields.io/azure-devops/coverage/blosc/caterva/1)
[![Documentation Status](https://readthedocs.org/projects/python-caterva/badge/?version=latest)](https://python-caterva.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Blosc/python-caterva/master?filepath=notebooks%2Fslicing-performance.ipynb)
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
pip install -r requirements.txt
```

### Compile

```sh
python setup.py build_ext --build-type=RelWithDebInfo
```

### Run tests

```sh
PYTHONPATH=. pytest
```

### Run bench

```sh
PYTHONPATH=. python bench/compare_getslice.py
```

### Installing

```sh
CFLAGS='' pip install caterva
```

We don't produce wheels yet, so you will currently need a C compiler in order to install python-caterva.  The reason why you need the `CFLAGS=''` above is to prevent Anaconda Python injecting their own paths for dependencies (LZ4, Zstd...).
