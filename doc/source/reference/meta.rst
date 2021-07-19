Metalayers
==========
Metalayers are small metadata for informing about the properties of data that is stored on a container. Caterva implements its own metalayer on top of C-Blosc2 for storing multidimensional information.

.. currentmodule:: caterva.meta

.. autoclass:: Meta
   :exclude-members: get, keys, items, values

.. currentmodule:: caterva.meta.Meta

Methods
-------

.. autosummary::
    :toctree: api/meta
    :nosignatures:

    __getitem__
    __setitem__
    get
    keys
    __iter__
    __contains__
