-----------------
Library Reference
-----------------

.. currentmodule:: cat4py

First level variables
=====================
.. py:attribute:: __version__
    The version of the cat4py package.

Utility functions (``cat4py.container``)
========================================
.. module:: cat4py.container
.. autofunction:: empty
.. autofunction:: from_buffer
.. autofunction:: from_numpy
.. autofunction:: from_file


The Container class
===================
.. module:: cat4py.container
.. autoclass:: Container
    :members: copy, get_metalayer, get_usermeta, has_metalayer, iter_read, iter_write, to_buffer, to_numpy, update_metalayer, update_usermeta
    :special-members: __init__, __getitem__, __setitem__

