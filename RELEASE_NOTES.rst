# Release notes for cat4py
| | |
| - | - |
| Author | The Blosc Developer Team |
| Contact | blosc@blosc.org |
| Date | 2019-09-17 |

## Changes from 0.3.0 to 0.3.1

  XXX version-specific blurb XXX


## Changes from 0.2.3 to 0.3.0

* Set the development status to alpha.

* Add instructions about installing cat4py from pip.

* `getitem` and `setitem` are now special methods in `ext.Container`.

* Add new class from numpy arrays `NPArray`.

* Support for serializing/deserializing Containers to/from serialized frames (bytes).

* The `pshape` is calculated automatically if is `None`.

* Add a `.sframe` attribute for the serialized frame.

* Big refactor for more consistent inheritance among classes.

* The `from_numpy()` function always return a `NPArray` now.


## Changes from 0.2.2 to 0.2.3

* Rename `MANINFEST.in` for `MANIFEST.in`.

* Fix the list of available cnames.


## Changes from 0.2.1 to 0.2.2

* Added a `MANIFEST.in` for including all C-Blosc2 and Caterva sources in package.


## Changes from 0.1.1 to 0.2.1

* Docstrings has been added. In addition, the documentation can be found at:
https://cat4py.readthedocs.io.

* Add a `copy` parameter to `from_file()`.

* `complib` has been renamed to `cname` for compatibility with blosc-powered packages.

* The use of an itemsize different than a 2 power is allowed now.

