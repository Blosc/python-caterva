Release notes
=============

Changes from 0.5.3 to 0.5.4
---------------------------

* Update caterva submodule to 0.5.0

Changes from 0.5.1 to 0.5.3
---------------------------

* Fix dependencies installation issue.

Changes from 0.5.0 to 0.5.1
---------------------------

* Update `setup.py` and add `pyproject.toml`.

Changes from 0.4.2 to 0.5.0
---------------------------

* Big c-core refactoring improving the slicing performance.
* Implement `__setitem__` method for arrays to allow to update the values of the arrays.
* Use Blosc special-constructors to initialize the arrays.
* Improve the buffer and array protocols.
* Remove the data type support in order to simplify the library.

Changes from 0.4.1 to 0.4.2
---------------------------

* Add files in `MANIFEST.in`.

Changes from 0.4.0 to 0.4.1
---------------------------

* Fix invalid values for classifiers defined in `setup.py`.

Changes from 0.3.0 to 0.4.0
---------------------------

* Compile the package using scikit-build.

* Introduce a second level of multidimensional chunking.

* Complete API renaming.

* Support the buffer protocol and the numpy array protocol.

* Generalize the slicing.

* Make cat4py independent of numpy.


Changes from 0.2.3 to 0.3.0
---------------------------

* Set the development status to alpha.

* Add instructions about installing cat4py from pip.

* `getitem` and `setitem` are now special methods in `ext.Container`.

* Add new class from numpy arrays `NPArray`.

* Support for serializing/deserializing Containers to/from serialized frames (bytes).

* The `pshape` is calculated automatically if is `None`.

* Add a `.sframe` attribute for the serialized frame.

* Big refactor for more consistent inheritance among classes.

* The `from_numpy()` function always return a `NPArray` now.


Changes from 0.2.2 to 0.2.3
---------------------------

* Rename `MANINFEST.in` for `MANIFEST.in`.

* Fix the list of available cnames.


Changes from 0.2.1 to 0.2.2
---------------------------

* Added a `MANIFEST.in` for including all C-Blosc2 and Caterva sources in package.


Changes from 0.1.1 to 0.2.1
---------------------------

* Docstrings has been added. In addition, the documentation can be found at:
  https://cat4py.readthedocs.io.

* Add a `copy` parameter to `from_file()`.

* `complib` has been renamed to `cname` for compatibility with blosc-powered packages.

* The use of an itemsize different than a 2 power is allowed now.
