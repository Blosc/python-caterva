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
$ python setup.py build_ext -i
```

Compiling the extension implies re-compiling C-Blosc2 and Caterva sources everytime, so a trick for accelerating the process is to direct the compiler to not optimize the code:

```sh
$ CFLAGS=-O0 python setup.py build_ext -i
```

### Run example

```sh
$ PYTHONPATH=. python examples/simple.py
```
