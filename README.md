[![Build Status](https://dev.azure.com/blosc/caterva/_apis/build/status/Blosc.cat4py?branchName=master)](https://dev.azure.com/blosc/caterva/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/Blosc/cat4py/branch/master/graph/badge.svg)](https://codecov.io/gh/Blosc/cat4py)
[![Documentation Status](https://readthedocs.org/projects/cat4py/badge/?version=latest)](https://cat4py.readthedocs.io/en/latest/?badge=latest)
# cat4py

Python wrapper for Caterva.  Still on development.

## Clone repo and submodules

```sh
$ git clone https://github.com/Blosc/cat4py
$ git submodule init
$ git submodule update --recursive --remote 
```

## Development workflow

### Compile

```sh
$ CFLAGS='' python setup.py build_ext -i
```

**Please note**: If the CFLAGS environment variable is not passed, Anaconda Python (maybe other distributions too) will inject their own paths there. As a result, it will find possible incompatible headers/libs for Blosc, LZ4 or Zstd.  We understand packagers trying to re-use shared libraries in their setups, but this can create issues when normal users try to compile extensions by themselves.

Compiling the extension implies re-compiling C-Blosc2 and Caterva sources everytime, so a trick for accelerating the process during the development process is to direct the compiler to not optimize the code, and use pre-installed Blosc2 and Caterva libraries:

```sh
$ CFLAGS=-O0 python setup.py build_ext -i --blosc2=/usr/local --caterva=/usr/local
```

### Run tests

```sh
$ PYTHONPATH=. pytest
```

### Run example

```sh
$ PYTHONPATH=. python examples/ex_persistency.py
```

### Installing

```sh
$ CFLAGS='' pip install cat4py
```

We don't produce wheels yet, so you will currently need a C compiler in order to install cat4py.  The reason why you need the `CFLAGS=''` above is to prevent Anaconda Python injecting their own paths for dependencies (LZ4, Zstd...). 
