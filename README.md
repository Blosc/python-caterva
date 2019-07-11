[![Build Status](https://dev.azure.com/blosc/caterva/_apis/build/status/Blosc.cat4py?branchName=master)](https://dev.azure.com/blosc/caterva/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/Blosc/cat4py/branch/master/graph/badge.svg)](https://codecov.io/gh/Blosc/cat4py)
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

**Please note**: The Anaconda python interpreter messes with CFLAGS if it is not set by the user.  If the CFLAGS is not passed, Anaconda python will inject their own paths, so it will find a possible `/Users/faltet/miniconda3/include/blosc.h`, which is not compatible with the blosc.h header for Blosc2.  I suppose the only solution long term will be to use `blosc2.h` and `libblosc2.so`.  Meanwhile, use CFLAGS explicitly so as to not mess with Anaconda python own business. 

Compiling the extension implies re-compiling C-Blosc2 and Caterva sources everytime, so a trick for accelerating the process is to direct the compiler to not optimize the code:

```sh
$ CFLAGS=-O0 python setup.py build_ext -i
```

### Run example

```sh
$ PYTHONPATH=. python examples/ex_persistency.py
```
