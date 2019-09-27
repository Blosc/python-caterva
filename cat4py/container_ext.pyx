# Hey Cython, this is Python 3!
# cython: language_level=3

#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
cimport numpy as np
import msgpack

from libc.stdlib cimport malloc, free
from libcpp cimport bool
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from collections import namedtuple

import os.path

cdef extern from "<stdint.h>":
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long long uint64_t


cdef extern from "blosc2.h":
    ctypedef enum:
        BLOSC_NOFILTER
        BLOSC_SHUFFLE
        BLOSC_BITSHUFFLE
        BLOSC_DELTA
        BLOSC_TRUNC_PREC
        BLOSC_BLOSCLZ
        BLOSC_LZ4
        BLOSC_LZ4HC
        BLOSC_ZLIB
        BLOSC_ZSTD
        BLOSC_LIZARD
        BLOSC2_MAX_FILTERS
        BLOSC2_MAX_METALAYERS
        BLOSC2_PREFILTER_INPUTS_MAX

    ctypedef struct blosc2_frame_metalayer
    ctypedef struct blosc2_frame:
      char* fname
      uint8_t* sdata
      uint8_t* coffsets
      int64_t len
      int64_t maxlen
      uint32_t trailer_len

    blosc2_frame *blosc2_new_frame(char *fname)

    ctypedef struct blosc2_context
    ctypedef int* blosc2_prefilter_fn

    ctypedef struct blosc2_schunk:
        uint8_t version
        uint8_t flags1
        uint8_t flags2
        uint8_t flags3
        uint8_t compcode
        uint8_t clevel
        int32_t typesize
        int32_t blocksize
        int32_t chunksize
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        int32_t nchunks
        int64_t nbytes
        int64_t cbytes
        uint8_t** data
        blosc2_frame* frame
        blosc2_context* cctx
        blosc2_context* dctx
        int16_t nmetalayers
        uint8_t* usermeta
        int32_t usermeta_len

    ctypedef struct blosc2_prefilter_params:
        int ninputs
        uint8_t* inputs[BLOSC2_PREFILTER_INPUTS_MAX]
        int32_t input_typesizes[BLOSC2_PREFILTER_INPUTS_MAX]
        void *user_data
        uint8_t *out
        int32_t out_size
        int32_t out_typesize

    ctypedef struct blosc2_cparams:
        uint8_t compcode
        uint8_t clevel
        int use_dict
        int32_t typesize
        int16_t nthreads
        int32_t blocksize
        void* schunk
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filters_meta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params *pparams

    ctypedef struct blosc2_dparams:
        int nthreads
        void* schunk

    blosc2_cparams BLOSC2_CPARAMS_DEFAULTS
    blosc2_dparams BLOSC2_DPARAMS_DEFAULTS

    int64_t blosc2_schunk_to_frame(blosc2_schunk* schunk, blosc2_frame* frame)
    int blosc2_has_metalayer(blosc2_schunk *schunk, char *name)
    int blosc2_add_metalayer(blosc2_schunk *schunk, char *name, uint8_t *content, uint32_t content_len)
    int blosc2_update_metalayer(blosc2_schunk *schunk, char *name, uint8_t *content, uint32_t content_len)
    int blosc2_get_metalayer(blosc2_schunk *schunk, char *name, uint8_t **content, uint32_t *content_len)

    int blosc2_update_usermeta(blosc2_schunk *schunk, uint8_t *content, int32_t content_len, blosc2_cparams cparams)
    int blosc2_get_usermeta(blosc2_schunk* schunk, uint8_t** content)

    char* blosc_list_compressors()

    int blosc2_free_frame(blosc2_frame *frame)


cdef extern from "caterva.h":
    ctypedef enum:
        CATERVA_MAXDIM

    ctypedef enum caterva_storage_t:
        CATERVA_STORAGE_BLOSC,
        CATERVA_STORAGE_PLAINBUFFER

    ctypedef struct caterva_ctx_t:
        void *(*alloc)(size_t)
        void (*free)(void *)
        blosc2_cparams cparams
        blosc2_dparams dparams

    ctypedef struct caterva_dims_t:
        int64_t dims[CATERVA_MAXDIM]
        int8_t ndim

    ctypedef caterva_dims_t CATERVA_DIMS_DEFAULTS

    ctypedef struct part_cache_s:
        uint8_t *data
        int32_t nchunk

    ctypedef struct caterva_array_t:
        caterva_ctx_t *ctx
        caterva_storage_t storage
        blosc2_schunk *sc
        uint8_t *buf
        int64_t shape[CATERVA_MAXDIM]
        int32_t pshape[CATERVA_MAXDIM]
        int64_t eshape[CATERVA_MAXDIM]
        int64_t size
        int32_t psize
        int64_t esize
        int8_t ndim
        bool empty
        bool filled
        int64_t nparts
        part_cache_s part_cache

    caterva_ctx_t *caterva_new_ctx(void *(*)(size_t), void (*free)(void *),
                                   blosc2_cparams cparams, blosc2_dparams dparams)
    int caterva_free_ctx(caterva_ctx_t *ctx)
    caterva_dims_t caterva_new_dims(int64_t *dims, int8_t ndim)
    caterva_array_t *caterva_empty_array(caterva_ctx_t *ctx, blosc2_frame *fr, caterva_dims_t *pshape)
    int caterva_free_array(caterva_array_t *carr)
    caterva_array_t *caterva_from_sframe(caterva_ctx_t *ctx, uint8_t *sframe, int64_t len, bool copy)
    caterva_array_t *caterva_from_file(caterva_ctx_t *ctx, const char *filename, bool copy)
    int caterva_from_buffer(caterva_array_t *dest, caterva_dims_t *shape, void *src)
    int caterva_to_buffer(caterva_array_t *src, void *dest)
    int caterva_get_slice(caterva_array_t *dest, caterva_array_t *src, caterva_dims_t *start, caterva_dims_t *stop)
    int caterva_repart(caterva_array_t *dest, caterva_array_t *src)
    int caterva_squeeze(caterva_array_t *src)
    int caterva_append(caterva_array_t *carr, void *part, int64_t partsize)
    int caterva_get_slice_buffer(void *dest, caterva_array_t *src, caterva_dims_t *start,
                                 caterva_dims_t *stop, caterva_dims_t *d_pshape)
    int caterva_get_slice_buffer_no_copy(void **dest, caterva_array_t *src, caterva_dims_t *start,
                                         caterva_dims_t *stop, caterva_dims_t *d_pshape)
    int caterva_set_slice_buffer(caterva_array_t *dest, void *src, caterva_dims_t *start, caterva_dims_t *stop)
    int caterva_update_shape(caterva_array_t *src, caterva_dims_t *shape)
    caterva_dims_t caterva_get_shape(caterva_array_t *src)
    caterva_dims_t caterva_get_pshape(caterva_array_t *src)
    int caterva_copy(caterva_array_t *dest, caterva_array_t *src)


# Codecs
BLOSCLZ = BLOSC_BLOSCLZ
LZ4 = BLOSC_LZ4
LZ4HC = BLOSC_LZ4HC
ZLIB = BLOSC_ZLIB
ZSTD = BLOSC_ZSTD
LIZARD = BLOSC_LIZARD

# Filters
NOFILTER = BLOSC_NOFILTER
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE
DELTA = BLOSC_DELTA
TRUNC_PREC = BLOSC_TRUNC_PREC

cnames = blosc_list_compressors()
# Build a dict with all the available cnames
_cnames2codecs = {
    "blosclz": BLOSCLZ,
    "lz4": LZ4,
    "lz4hc": LZ4HC,
    "zlib": ZLIB,
    "zstd": ZSTD,
    "lizard": LIZARD,
}
cnames2codecs = {}
blosc_cnames = blosc_list_compressors()
blosc_cnames = blosc_cnames.split(b",")
blosc_cnames = [cname.decode() for cname in blosc_cnames]
for cname in _cnames2codecs:
    if cname in blosc_cnames:
        cnames2codecs[cname] = _cnames2codecs[cname]

# Defaults for compression params
cparams_dflts = {
    'itemsize': 4,
    'cname': 'lz4',
    'clevel': 5,
    'use_dict': False,
    'cnthreads': 1,
    'dnthreads': 1,
    'blocksize': 0,
    'filters': [BLOSC_SHUFFLE],
    'filters_meta': [0],  # no actual meta info for SHUFFLE, but anyway...
    }


cdef class CParams:
    cdef str cname
    cdef uint8_t compcode
    cdef uint8_t clevel
    cdef int use_dict
    cdef int32_t itemsize
    cdef int16_t nthreads
    cdef int32_t blocksize
    cdef void* schunk
    cdef uint8_t filters[BLOSC2_MAX_FILTERS]
    cdef uint8_t filters_meta[BLOSC2_MAX_FILTERS]
    cdef blosc2_prefilter_fn prefilter
    cdef blosc2_prefilter_params* pparams

    def __init__(self, **kargs):
        cname = kargs.get('cname', cparams_dflts['cname'])
        if isinstance(cname, bytes):
            cname = cname.decode()
        if cname not in cnames2codecs:
            raise ValueError(f"'{cname}' is not among the list of available codecs ({cnames2codecs.keys()})")
        self.cname = cname
        self.compcode = cnames2codecs[cname]
        self.clevel = kargs.get('clevel', cparams_dflts['clevel'])
        self.itemsize = kargs.get('itemsize', cparams_dflts['itemsize'])
        self.use_dict = kargs.get('use_dict', cparams_dflts['use_dict'])
        self.nthreads = kargs.get('cnthreads', cparams_dflts['cnthreads'])
        self.blocksize = kargs.get('blocksize', cparams_dflts['blocksize'])
        self.prefilter = NULL  # TODO: implement support for prefilters
        self.pparams = NULL    # TODO: implement support for prefilters

        # Filter pipeline
        for i in range(BLOSC2_MAX_FILTERS):
            self.filters[i] = 0
        for i in range(BLOSC2_MAX_FILTERS):
            self.filters_meta[i] = 0

        filters = kargs.get('filters', cparams_dflts['filters'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters), BLOSC2_MAX_FILTERS):
            self.filters[i] = filters[i - BLOSC2_MAX_FILTERS + len(filters)]

        filters_meta = kargs.get('filters_meta', cparams_dflts['filters_meta'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters_meta), BLOSC2_MAX_FILTERS):
            self.filters_meta[i] = filters_meta[i - BLOSC2_MAX_FILTERS + len(filters_meta)]


cdef class DParams:
    cdef int nthreads
    cdef void* schunk

    def __init__(self, **kargs):
        self.nthreads = kargs.get('dnthreads', cparams_dflts['dnthreads'])


cdef class Context:
    cdef caterva_ctx_t *_ctx
    cdef CParams cparams
    cdef DParams dparams

    def __init__(self, CParams cparams=None, DParams dparams=None):
        cdef blosc2_cparams _cparams
        if cparams is None:
            cparams = CParams()
        self.cparams = cparams
        _cparams.typesize = cparams.itemsize  # TODO: typesize -> itemsize in c-blosc2
        _cparams.compcode = cparams.compcode
        _cparams.clevel = cparams.clevel
        _cparams.use_dict = int(cparams.use_dict)
        _cparams.nthreads = cparams.nthreads
        _cparams.blocksize = cparams.blocksize
        _cparams.prefilter = cparams.prefilter
        _cparams.pparams = cparams.pparams
        for i in range(BLOSC2_MAX_FILTERS):
            _cparams.filters[i] = cparams.filters[i]
        for i in range(BLOSC2_MAX_FILTERS):
            _cparams.filters_meta[i] = cparams.filters_meta[i]
        cdef blosc2_dparams _dparams
        if dparams is None:
            dparams=DParams()
        self.dparams = dparams
        _dparams.nthreads = dparams.nthreads
        self._ctx = caterva_new_ctx(NULL, NULL, _cparams, _dparams)

    def __dealloc__(self):
        caterva_free_ctx(self._ctx)

    def tocapsule(self):
        return PyCapsule_New(self._ctx, "caterva_ctx_t*", NULL)


cdef class WriteIter:
    cdef Container arr
    cdef buffer
    cdef dtype
    cdef buffer_shape
    cdef int32_t buffer_len
    cdef int32_t part_len

    # TODO: is np.dtype really necessary here?  Container does not have this notion, so...
    def __init__(self, arr):
        self.arr = arr
        self.dtype = np.dtype(f"S{arr.itemsize}")
        self.part_len = self.arr.array.psize * self.arr.itemsize

    def __iter__(self):
        self.buffer_shape = None
        self.buffer_len = 0
        self.buffer = None
        self.memview = None
        return self

    def __next__(self):
        cdef char* data_pointer
        if self.buffer is not None:
            if self.part_len != self.buffer_len:
                # Extended partition; pad with zeros
                item = np.frombuffer(self.memview[:self.buffer_len], self.dtype).reshape(self.buffer_shape)
                item = np.pad(item, [(0, self.arr.array.pshape[i] - item.shape[i]) for i in range(self.arr.ndim)],
                              mode='constant', constant_values=0)
                item = item.tobytes()
                data_pointer = <char*> item
            else:
                data_pointer = <char*> self.buffer
            caterva_append(self.arr.array, data_pointer, self.part_len)

        if self.arr.array.filled:
            raise StopIteration

        aux = [self.arr.array.eshape[i] // self.arr.array.pshape[i] for i in range(self.arr.array.ndim)]
        start_ = [0 for _ in range(self.arr.array.ndim)]
        inc = 1
        for i in range(self.arr.array.ndim - 1, -1, -1):
            start_[i] = self.arr.array.nparts % (aux[i] * inc) // inc
            start_[i] *= self.arr.array.pshape[i]
            inc *= aux[i]

        stop_ = [start_[i] + self.arr.array.pshape[i] for i in range(self.arr.array.ndim)]
        for i in range(self.arr.array.ndim):
            if stop_[i] > self.arr.array.shape[i]:
                stop_[i] = self.arr.array.shape[i]

        sl = tuple([slice(start_[i], stop_[i]) for i in range(self.arr.array.ndim)])
        shape = [s.stop - s.start for s in sl]
        IterInfo = namedtuple("IterInfo", "slice, shape, size")
        info = IterInfo(slice=sl, shape=shape, size=np.prod(shape))

        # Allocate a new buffer if needed
        self.buffer_shape = shape
        self.buffer_len = np.prod(shape) * self.arr.itemsize
        if self.buffer is None:
            self.buffer = bytearray(self.part_len)
            self.memview = memoryview(self.buffer)
        return self.memview[:self.buffer_len], info


cdef class ReadIter:
    cdef Container arr
    cdef blockshape
    cdef dtype
    cdef nparts
    cdef object IterInfo

    def __init__(self, arr, blockshape):
        if not arr.filled:
            raise ValueError("Container is not filled")
        self.arr = arr
        if blockshape is None:
            blockshape = arr.pshape
        self.blockshape = blockshape
        self.nparts = 0
        self.IterInfo = namedtuple("IterInfo", "slice, shape, size")

    def __iter__(self):
        return self

    def __next__(self):
        ndim = self.arr.ndim
        shape = tuple(self.arr.shape)
        eshape = [0 for i in range(ndim)]
        for i in range(ndim):
            if shape[i] % self.blockshape[i] == 0:
                eshape[i] = self.blockshape[i] * (shape[i] // self.blockshape[i])
            else:
                eshape[i] = self.blockshape[i] * (shape[i] // self.blockshape[i] + 1)
        aux = [eshape[i] // self.blockshape[i] for i in range(ndim)]
        if self.nparts >= np.prod(aux):
            raise StopIteration

        start_ = [0 for _ in range(ndim)]
        inc = 1
        for i in range(ndim - 1, -1, -1):
            start_[i] = self.nparts % (aux[i] * inc) // inc
            start_[i] *= self.blockshape[i]
            inc *= aux[i]

        stop_ = [start_[i] + self.blockshape[i] for i in range(ndim)]
        for i in range(ndim):
            if stop_[i] > shape[i]:
                stop_[i] = shape[i]

        sl = tuple([slice(start_[i], stop_[i]) for i in range(ndim)])
        sh = [s.stop - s.start for s in sl]
        info = self.IterInfo(slice=sl, shape=sh, size=np.prod(sh))
        self.nparts += 1

        buf = self.arr.slicebuffer(info.slice)
        return buf, info


cdef get_caterva_shape(shape):
    ndim = len(shape)
    cdef int64_t *shape_ = <int64_t*>malloc(ndim * sizeof(int64_t))
    for i in range(ndim):
        shape_[i] = shape[i]
    cdef caterva_dims_t _shape = caterva_new_dims(shape_, ndim)
    free(shape_)
    return _shape


cdef get_caterva_start_stop(ndim, key, shape):
    start = [s.start if s.start is not None else 0 for s in key]
    stop = [s.stop if s.stop is not None else sh for s, sh in zip(key, shape)]
    start_ = <int64_t*> malloc(ndim * sizeof(int64_t))
    for i in range(ndim):
        start_[i] = start[i]
    cdef caterva_dims_t _start = caterva_new_dims(start_, ndim)
    free(start_)
    stop_ = <int64_t*> malloc(ndim * sizeof(int64_t))
    for i in range(ndim):
        stop_[i] = stop[i]
    cdef caterva_dims_t _stop = caterva_new_dims(stop_, ndim)
    free(stop_)
    pshape_ = <int64_t*> malloc(ndim * sizeof(int64_t))
    for i in range(ndim):
        pshape_[i] = stop[i] - start[i]
    cdef caterva_dims_t _pshape = caterva_new_dims(pshape_, ndim)
    free(pshape_)
    size = np.prod([stop[i] - start[i] for i in range(ndim)])

    return _start, _stop, _pshape, size


cdef class Container:
    cdef Context ctx
    cdef caterva_array_t *array
    cdef kargs
    cdef usermeta_len
    cdef CParams cparams
    cdef DParams dparams

    @property
    def shape(self):
        """The shape of this container."""
        cdef caterva_dims_t shape = caterva_get_shape(self.array)
        return tuple([shape.dims[i] for i in range(shape.ndim)])

    @property
    def pshape(self):
        """The partition shape of this container."""
        if self.array.storage == CATERVA_STORAGE_PLAINBUFFER:
            return None
        cdef caterva_dims_t pshape = caterva_get_pshape(self.array)
        return tuple([pshape.dims[i] for i in range(pshape.ndim)])

    @property
    def cratio(self):
        """The compression ratio for this container."""
        if self.array.storage is not CATERVA_STORAGE_BLOSC:
            return 1
        return self.array.sc.nbytes / self.array.sc.cbytes

    @property
    def itemsize(self):
        """The itemsize of this container."""
        return self.array.ctx.cparams.typesize

    @property
    def clevel(self):
        """The compression level for this container."""
        return self.array.ctx.cparams.clevel

    @property
    def cname(self):
        """The compression codec name for this container."""
        return self.cparams.cname

    @property
    def filters(self):
        """The filters list for this container."""
        return [self.array.ctx.cparams.filters[i] for i in range(BLOSC2_MAX_FILTERS)]

    @property
    def size(self):
        """The size (in items) for this container."""
        return self.array.size

    @property
    def psize(self):
        """The partition size (in items) for this container."""
        return self.array.psize

    @property
    def npart(self):
        """The number of partitions in this container."""
        return int(self.array.esize / self.array.psize)

    @property
    def ndim(self):
        """The number of dimensions of this container."""
        return self.array.ndim

    @property
    def filled(self):
        """Whether the container is filled or not."""
        return self.array.filled

    def __init__(self, **kwargs):
        self.cparams = CParams(**kwargs)
        self.dparams = DParams(**kwargs)
        self.ctx = Context(self.cparams, self.dparams)
        cdef caterva_ctx_t * ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(self.ctx.tocapsule(), "caterva_ctx_t*")
        self.usermeta_len = 0

        cdef int64_t *pshape_
        cdef caterva_dims_t _pshape
        cdef blosc2_frame *_frame

        pshape = kwargs['pshape'] if 'pshape' in kwargs else None
        filename = kwargs['filename'] if 'filename' in kwargs else None
        memframe = kwargs['memframe'] if 'memframe' in kwargs else None
        if filename is not None and memframe is True:
            raise ValueError("You cannot specify a `filename` and set `memframe` to True at once.")
        if pshape is None:
            if filename is not None:
                raise NotImplementedError
            else:
                # We are probably de-serializing
                self.array = caterva_empty_array(ctx_, NULL, NULL)
        else:
            ndim = len(pshape)
            pshape_ = <int64_t*> malloc(ndim * sizeof(int64_t))
            for i in range(ndim):
                pshape_[i] = pshape[i]
            _pshape = caterva_new_dims(pshape_, ndim)
            free(pshape_)

            if filename is None:
                if memframe:
                    _frame = blosc2_new_frame(NULL)
                else:
                    _frame = NULL
            else:
                if os.path.isfile(filename):
                    raise FileExistsError
                else:
                    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
                    _frame = blosc2_new_frame(filename)

            self.array = caterva_empty_array(ctx_, _frame, &_pshape)

        if 'metalayers' in kwargs:
            metalayers = kwargs['metalayers']
            for name, content in metalayers.items():
                name = name.encode("utf-8") if isinstance(name, str) else name
                content = msgpack.packb(content)
                blosc2_add_metalayer(self.array.sc, name, content, len(content))

    def __getitem__(self, key):
        cdef caterva_dims_t _start, _stop, _pshape
        ndim = self.array.ndim
        _start, _stop, _pshape, size = get_caterva_start_stop(ndim, key, self.shape)
        bsize = size * self.itemsize
        buffer = bytes(bsize)
        err = caterva_get_slice_buffer(<char *> buffer, self.array, &_start, &_stop, &_pshape)
        return buffer

    def tocapsule(self):
        return PyCapsule_New(self.array, "caterva_array_t*", NULL)

    def slicebuffer(self, key):
        key = list(key)
        for i, sl in enumerate(key):
            if type(sl) is not slice:
                key[i] = slice(sl, sl+1, None)

        cdef caterva_dims_t _start, _stop, _pshape
        ndim = self.array.ndim
        _start, _stop, _pshape, size = get_caterva_start_stop(ndim, key, self.shape)

        bsize = size * self.itemsize
        buffer = bytes(bsize)
        caterva_get_slice_buffer(<char *> buffer, self.array, &_start, &_stop, &_pshape)

        return buffer

    def updateshape(self, shape):
        cdef caterva_dims_t _shape = get_caterva_shape(shape)
        caterva_update_shape(self.array, &_shape)

    def squeeze(self):
        caterva_squeeze(self.array)

    def to_buffer(self):
        size = self.size * self.itemsize
        buffer = bytes(size)
        caterva_to_buffer(self.array, <void *> <char *> buffer)
        return buffer

    def to_sframe(self):
        if not self.array.filled:
            raise NotImplementedError("The Container is not filled")
        if self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("The Container is backed by a plain buffer")
        cdef char* fname
        cdef char* data
        cdef bytes sdata
        cdef blosc2_frame* frame
        if self.array.sc.frame != NULL:
            fname = self.array.sc.frame.fname
            if fname == NULL:
                data = <char*> self.array.sc.frame.sdata
                sdata = data[:self.array.sc.frame.len]
            else:
                with open(fname, 'rb') as f:
                    sdata = f.read()
        else:
            # Container is not backed by a frame, so create a new one and fill it
            # Here there is a double copy; how to avoid it?
            frame = blosc2_new_frame(NULL)
            blosc2_schunk_to_frame(self.array.sc, frame)
            data = <char*> frame.sdata
            sdata = data[:frame.len]
            blosc2_free_frame(frame)
        return sdata

    def copy(self, Container dest):
        caterva_copy(dest.array, self.array)

    def has_metalayer(self, name):
        if  self.array.storage != CATERVA_STORAGE_BLOSC and self.array.sc.frame == NULL:
            return NotImplementedError
        name = name.encode("utf-8") if isinstance(name, str) else name
        n = blosc2_has_metalayer(self.array.sc, name)
        return False if n < 0 else True

    def get_metalayer(self, name):
        if  self.array.storage != CATERVA_STORAGE_BLOSC and self.array.sc.frame == NULL:
            return NotImplementedError
        name = name.encode("utf-8") if isinstance(name, str) else name
        cdef uint8_t *_content
        cdef uint32_t content_len
        n = blosc2_get_metalayer(self.array.sc, name, &_content, &content_len)
        content = <char *>_content
        content = content[:content_len]  # does a copy
        free(_content)
        return content

    def update_metalayer(self, name, content):
        name = name.encode("utf-8") if isinstance(name, str) else name
        content_ = self.get_metalayer(name)
        if len(msgpack.packb(content_)) != len(content):
            return ValueError("The length of the content in a metalayer cannot change.")
        n = blosc2_update_metalayer(self.array.sc, name, content, len(content))
        return n

    def update_usermeta(self, content):
        n = blosc2_update_usermeta(self.array.sc, content, len(content), BLOSC2_CPARAMS_DEFAULTS)
        self.usermeta_len = len(content)
        return n

    def get_usermeta(self):
        cdef uint8_t *_content
        n = blosc2_get_usermeta(self.array.sc, &_content)
        if n < 0:
            raise ValueError("Cannot get the usermeta section")
        content = <char *>_content
        content = content[:n]  # does a copy
        free(_content)
        return content

    def __dealloc__(self):
        if self.array != NULL:
            caterva_free_array(self.array)


def from_file(Container arr, filename, copy):
    ctx = Context()
    cdef caterva_ctx_t * ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(ctx.tocapsule(), "caterva_ctx_t*")
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef caterva_array_t *a_ = caterva_from_file(ctx_, filename, copy)
    arr.ctx = ctx
    arr.array = a_


def from_sframe(Container arr, bytes sframe, copy):
    ctx = Context()
    cdef caterva_ctx_t *ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(ctx.tocapsule(), "caterva_ctx_t*")
    cdef uint8_t *frame_ = sframe
    cdef caterva_array_t *a_ = caterva_from_sframe(ctx_, frame_, len(sframe), copy)
    arr.ctx = ctx
    arr.array = a_


def from_buffer(Container arr, shape, buf):
    if arr.pshape is not None:
        assert(len(shape) == len(arr.pshape))

    cdef caterva_dims_t _shape = get_caterva_shape(shape)
    cdef int retcode = caterva_from_buffer(arr.array, &_shape, <void*> <char *> buf)
    if retcode < 0:
        raise ValueError("Error filling the caterva object with buffer")


def list_cnames():
    """Return a list of all the available compressor names.

    Returns
    -------
    List
        A list with all the available compressor names.
    """
    cnames = blosc_list_compressors()
    cnames = cnames.split(b",")
    return cnames
