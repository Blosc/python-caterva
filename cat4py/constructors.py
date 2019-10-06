import numpy as np
from . import container_ext as ext
from .container import Container
from .tlarray import TLArray
from .nparray import NPArray
from .container import get_pshape_guess


def update_kwargs(shape, dtype, kwargs):
    """Compute some decent guesses for params not in `kwargs`."""
    if "pshape" not in kwargs or kwargs["pshape"] is None:
        if dtype is not None:
            dtype = np.dtype(dtype)
            itemsize = dtype.itemsize
            kwargs["itemsize"] = itemsize
        elif "itemsize" in kwargs:
            itemsize = kwargs["itemsize"]
        else:
            itemsize = ext.cparams_dflts["itemsize"]
        kwargs["pshape"] = get_pshape_guess(shape, itemsize)
    return kwargs


def empty(shape, dtype=None, **kwargs):
    """Create an empty container.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final container.
    dtype: str or numpy.dtype
        The dtype of the data.  Default: None.

    Returns
    -------
    TLArray or NPArray
        If `dtype` is None, a new :py:class:`TLArray` object is returned.
        If `dtype` is not None, a new :py:class:`NPArray` is returned.
    """
    kwargs = update_kwargs(shape, dtype, kwargs)

    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    arr.updateshape(shape)
    return arr


def from_buffer(buffer, shape, dtype=None, **kwargs):
    """Create a container out of a buffer.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.
    dtype: numpy.dtype
        The dtype of the data.  Default: None.

    Returns
    -------
    TLArray or NPArray
        If `dtype` is None, a new :py:class:`TLArray` object is returned.
        If `dtype` is not None, a new :py:class:`NPArray` is returned.
    """
    kwargs = update_kwargs(shape, dtype, kwargs)
    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    ext.from_buffer(arr, shape, buffer)
    return arr


def from_numpy(nparray, **kwargs):
    """Create a NPArray container out of a NumPy array.

    In addition to regular arguments, you can pass any keyword argument that
    is supported by the :py:meth:`Container.__init__` constructor.

    Parameters
    ----------
    nparray: numpy.array
        The NumPy array to populate the container with.

    Returns
    -------
    NPArray
        The new :py:class:`NPArray` object.
    """
    kwargs = update_kwargs(nparray.shape, nparray.dtype, kwargs)
    arr = from_buffer(bytes(nparray), nparray.shape, dtype=nparray.dtype, **kwargs)
    return arr


def from_file(filename, copy=False):
    """Open a new container from `filename`.

    Parameters
    ----------
    filename: str
        The file having a Blosc2 frame format with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an on-disk, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    TLArray or NPArray
    """

    arr = Container()
    ext.from_file(arr, filename, copy)
    if arr.has_metalayer("numpy"):
        arr = NPArray.cast(arr)
        dtype = arr.get_metalayer("numpy")[b'dtype']
        arr.pre_init(dtype)
    else:
        arr = TLArray.cast(arr)
        arr.pre_init()

    return arr


def from_sframe(sframe, copy=False):
    """Open a new container from `sframe`.

    Parameters
    ----------
    sframe: bytes
        The Blosc2 serialized frame with a Caterva metalayer on it.
    copy: bool
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an in-memory, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    TLArray or NPArray
    """
    arr = Container()
    ext.from_sframe(arr, sframe, copy)
    if arr.has_metalayer("numpy"):
        arr = NPArray.cast(arr)
        dtype = arr.get_metalayer("numpy")[b'dtype']
        arr.pre_init(dtype)
    else:
        arr = TLArray.cast(arr)
        arr.pre_init()

    return arr
