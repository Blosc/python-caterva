# Release notes for python-blosc

:Author: The Blosc Developer Team
:Contact: blosc@blosc.org
:Date: 2019-09-17


## Changes from 0.2.1 to 0.2.2

* Added a `MANIFEST.in` for including all C-Blosc2 and Caterva sources in package.


## Changes from 0.1.1 to 0.2.1

* Docstrings has been added. In addition, the documentation can be found at:
https://cat4py.readthedocs.io.

* Add a `copy` parameter to `from_file()`.

* `complib` has been renamed to `cname` for compatibility with blosc-powered packages.

* The use of an itemsize different than a 2 power is allowed now.

