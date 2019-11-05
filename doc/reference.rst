-----------------
Library Reference
-----------------

.. currentmodule:: cat4py

First level variables
=====================
.. py:attribute:: __version__

    The version of the cat4py package.

.. py:attribute:: cnames

    List of available compression library names.

The Container class
===================
The low-level, multidimensional and type-less data container.

.. autoclass:: cat4py.Container
    :members:
    :exclude-members: updateshape
    :special-members: __init__, __getitem__, __setitem__

The TLArray class
=================
The basic, multidimensional and type-less object that inherits from the :py:class:`Container` class.

.. autoclass:: cat4py.TLArray
    :members:
    :exclude-members: updateshape, pre_init, cast
    :special-members: __init__, __getitem__, __setitem__

The NPArray class
=================
The multidimensional data container that plays well with NumPy.  Inherits from the :py:class:`Container` class.

.. autoclass:: cat4py.NPArray
    :members:
    :exclude-members: updateshape, pre_init, cast
    :special-members: __init__, __getitem__, __setitem__


Container constructors
======================
.. autofunction:: empty
.. autofunction:: from_buffer
.. autofunction:: from_numpy
.. autofunction:: from_file
.. autofunction:: from_sframe
