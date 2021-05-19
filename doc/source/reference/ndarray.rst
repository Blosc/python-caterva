NDArray
=======

The multidimensional data array class.

.. currentmodule:: cat4py.NDArray


Attributes
----------

.. autosummary::
    :toctree: api/ndarray

    itemsize
    ndim
    shape
    chunks
    blocks

Methods
-------

Slicing
+++++++

.. autosummary::
    :toctree: api/ndarray
    :nosignatures:

    __getitem__
    __setitem__
    slice

Metalayers
++++++++++

.. autosummary::
    :toctree: api/ndarray
    :nosignatures:

    has_meta
    get_meta
    update_meta
