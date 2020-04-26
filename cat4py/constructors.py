import numpy as np
from . import container_ext as ext
from .container import Container
from .tlarray import TLArray
from .nparray import NPArray


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

    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    kwargs = arr.kwargs
    ext.empty(arr, shape, **kwargs)
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
    arr = TLArray(**kwargs) if dtype is None else NPArray(dtype, **kwargs)
    kwargs = arr.kwargs

    ext.from_buffer(arr, buffer, shape, **kwargs)
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


def from_sframe(sframe, copy=False, **kwargs):
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
    ext.from_sframe(arr, sframe, copy, **kwargs)
    if arr.has_metalayer("numpy"):
        arr = NPArray.cast(arr)
        dtype = arr.get_metalayer("numpy")[b'dtype']
        arr.pre_init(dtype)
    else:
        arr = TLArray.cast(arr)
        arr.pre_init()

    return arr


def from_array(array, **kwargs):
    array_interface = array.__array_interface__
    if array_interface["strides"] is not None:
        raise NotImplementedError
    dtype = np.dtype(array_interface["typestr"])
    arr = NPArray(dtype, **kwargs)
    kwargs = arr.kwargs
    if "chunkshape" not in kwargs or kwargs["chunkshape"] is None:
        ext.from_array(arr, array_interface, **kwargs)
    else:
        ext.from_buffer(arr, bytes(array), array.shape, **kwargs)
    return arr
