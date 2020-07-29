from . import container_ext as ext
from .ndarray import NDArray
from .ndtarray import NDTArray


def empty(shape, itemsize, dtype=None, **kwargs):
    """Create an empty array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    itemsize: int
        The size, in bytes, of each element.
    dtype: str, optional
        The dtype of the data. (Default: None)

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:

            chunkshape: iterable object or None
                The chunk shape.  If `None`, the array is stored using a non-compressed buffer.
                (Default `None`)
            blockshape: iterable object or None
                The block shape.  If `None`, the array is stored using a non-compressed buffer.
                (Default `None`)
            filename: str or None
                The name of the file to store data.  If `None`, data is stored in-memory.
                (Default `None`)
            memframe: bool
                If True, the array is backed by a frame in-memory.  Else, by a super-chunk.
                (Default: `False`)
            metalayers: dict or None
                A dictionary with different metalayers.  One entry per metalayer:

                    key: bytes or str
                        The name of the metalayer.
                    value: object
                        The metalayer object that will be (de-)serialized using msgpack.

            cname: string
                The name for the compressor codec.  (Default: `"lz4"`)
            clevel: int (0 to 9)
                The compression level.  0 means no compression, and 9 maximum compression.
                (Default: `5`)
            filters: list
                The filter pipeline.  (Default: `[cat4py.SHUFFLE]`)
            filtersmeta: list
                The meta info for each filter in pipeline. (Default: `[0]`)
            nthreads: int
                The number of threads.  (Default: `1`)
            usedict: bool
                If a dictionary should be used during compression.  (Default: `False`)

    Returns
    -------
    out: NDArray or NDTArray
        If `dtype` is `None`, a `NDArray` with initialized data is returned.
        Else, a `NDTArray` is returned.
    """

    arr = NDArray(**kwargs) if dtype is None else NDTArray(dtype, **kwargs)
    kwargs = arr.kwargs
    ext.empty(arr, shape, itemsize, **kwargs)
    return arr


def from_buffer(buffer, shape, itemsize, dtype=None, **kwargs):
    """Create an array out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.
    itemsize: int
        The size, in bytes, of each element.
    dtype: str, optional
        The dtype of the data.  Default: None.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

    Returns
    -------
    out: NDArray or NDTArray
        If `dtype` is `None`, a `NDArray` is returned. Else, a `NDTArray` is returned.
    """
    arr = NDArray(**kwargs) if dtype is None else NDTArray(dtype, **kwargs)
    kwargs = arr.kwargs

    ext.from_buffer(arr, buffer, shape, itemsize, **kwargs)
    return arr


def copy(array, **kwargs):
    """Create a copy of an array.

    Parameters
    ----------
    array: NDArray or NDTArray
        The array to be copied.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

    Returns
    -------
    out: NDArray or NDTArray
        Depending on the source array class, a `NDArray` or a `NDTArray` is returned with a copy
        of the data.
    """
    arr = NDArray(**kwargs) if array._dtype is None else NDTArray(array._dtype, **kwargs)
    kwargs = arr.kwargs

    ext.copy(arr, array, **kwargs)

    return arr


def from_file(filename, copy=False):
    """Open a new container from `filename`.

    Parameters
    ----------
    filename: str
        The file having a Blosc2 frame format with a Caterva metalayer on it.
    copy: bool, optional
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an on-disk, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    out: NDArray or NDTArray
        If the file has a metalayer storing the type, a new `NDTArray` is returned.
        Else, a `NDArray` is returned.
    """

    arr = NDArray()
    ext.from_file(arr, filename, copy)
    if arr.has_metalayer("type"):
        arr = NDTArray.cast(arr)
        dtype = arr.get_metalayer("type")[b'dtype']
        arr.pre_init(dtype)
    else:
        arr = NDArray.cast(arr)
        arr.pre_init()

    return arr


def from_sframe(sframe, copy=False):
    """Open a new container from `sframe`.

    Parameters
    ----------
    sframe: bytes
        The Blosc2 serialized frame with a Caterva metalayer on it.
    copy: bool, optional
        If true, the container is backed by a new, sparse in-memory super-chunk.
        Else, an in-memory, frame-backed one is created (i.e. no copies are made).

    Returns
    -------
    out: NDArray or NDTArray
        If the serialized frame has a metalayer storing the type, a `NDTArray` is returned.
        Else, a `NDArray` is returned.
    """
    arr = NDArray()
    ext.from_sframe(arr, sframe, copy, **arr.kwargs)
    if arr.has_metalayer("type"):
        arr = NDTArray.cast(arr)
        dtype = arr.get_metalayer("type")[b'dtype']
        arr.pre_init(dtype)
    else:
        arr = NDArray.cast(arr)
        arr.pre_init()

    return arr


def asarray(ndarray, **kwargs):
    """Convert the input to an array.

    Parameters
    ----------
    array: array_like
        An array supporting the python buffer protocol and the numpy array interface.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`cat4py.empty` constructor.

    Returns
    -------
    out: NDTArray
        A Caterva array interpretation of `ndarray`. No copy is performed.
    """
    interface = ndarray.__array_interface__

    arr = NDTArray(interface["typestr"], **kwargs)
    kwargs = arr.kwargs

    ext.asarray(arr, ndarray, **kwargs)

    return arr
