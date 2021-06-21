from . import caterva_ext as ext
from .ndarray import NDArray


def empty(shape, itemsize, **kwargs):
    """Create an empty array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    itemsize: int
        The size, in bytes, of each element.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:

            chunks: iterable object or None
                The chunk shape.  If `None`, the array is stored using a non-compressed buffer.
                (Default `None`)
            blocks: iterable object or None
                The block shape.  If `None`, the array is stored using a non-compressed buffer.
                (Default `None`)
            filename: str or None
                The name of the file to store data.  If `None`, data is stored in-memory.
                (Default `None`)
            memframe: bool
                If True, the array is backed by a frame in-memory.  Else, by a super-chunk.
                (Default: `False`)
            meta: dict or None
                A dictionary with different metalayers.  One entry per metalayer:

                    key: bytes or str
                        The name of the metalayer.
                    value: object
                        The metalayer object that will be (de-)serialized using msgpack.

            codec: :py:class:`Codec`
                The name for the compressor codec.  (Default: :py:attr:`Codec.LZ4`)
            clevel: int (0 to 9)
                The compression level.  0 means no compression, and 9 maximum compression.
                (Default: 5)
            filters: list
                The filter pipeline.  (Default: [:py:attr:`Filter.SHUFFLE`])
            filtersmeta: list
                The meta info for each filter in pipeline. (Default: [0])
            nthreads: int
                The number of threads.  (Default: 1)
            usedict: bool
                If a dictionary should be used during compression.  (Default: False)

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs
    ext.empty(arr, shape, itemsize, **kwargs)
    return arr


def zeros(shape, itemsize, **kwargs):
    """Create an array, with zero being used as the default value
    for uninitialized portions of the array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    itemsize: int
        The size, in bytes, of each element.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs
    ext.zeros(arr, shape, itemsize, **kwargs)
    return arr


def full(shape, fill_value, **kwargs):
    """Create an array, with @p fill_value being used as the default value
    for uninitialized portions of the array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array..
    fill_value: bytes
        Default value to use for uninitialized portions of the array.
    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs
    ext.full(arr, shape, fill_value, **kwargs)
    return arr


def from_buffer(buffer, shape, itemsize, **kwargs):
    """Create an array out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.
    itemsize: int
        The size, in bytes, of each element.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs

    ext.from_buffer(arr, buffer, shape, itemsize, **kwargs)
    return arr


def copy(array, **kwargs):
    """Create a copy of an array.

    Parameters
    ----------
    array: NDArray
        The array to be copied.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` with a copy of the data.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs

    ext.copy(arr, array, **kwargs)

    return arr


def open(urlpath):
    """Open a new container from `urlpath`.

    .. warning:: Only one handler is supported per file.

    Parameters
    ----------
    urlpath: str
        The file having a Blosc2 frame format with a Caterva metalayer on it.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """

    arr = NDArray()
    ext.from_file(arr, urlpath)

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
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A Caterva array interpretation of `ndarray`. No copy is performed.
    """
    arr = NDArray(**kwargs)
    kwargs = arr.kwargs

    ext.asarray(arr, ndarray, **kwargs)

    return arr
