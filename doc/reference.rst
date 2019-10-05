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
.. autoclass:: cat4py.Container
    :members:
    :exclude-members: updateshape, slicebuffer
    :special-members: __init__, __getitem__, __setitem__


Container constructors
======================
.. autofunction:: empty
.. autofunction:: from_buffer
.. autofunction:: from_numpy
.. autofunction:: from_file
