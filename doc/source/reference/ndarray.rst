NDArray
=======

The multidimensional data array class. This class consists of a set of useful parameters and methods that allow not only to define an array correctly, but also to handle it in a simple way, being able to extract multidimensional slices from it.

.. currentmodule:: caterva.NDArray

Attributes
----------

.. autosummary::
    :toctree: api/ndarray

    itemsize
    ndim
    shape
    chunks
    blocks
    meta

Methods
-------

.. autosummary::
    :toctree: api/ndarray
    :nosignatures:

    __getitem__
    __setitem__
    slice
    resize
