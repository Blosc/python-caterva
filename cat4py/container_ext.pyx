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
from libc.stdint cimport uintptr_t
from libc.string cimport strdup
from .container import Container as HLContainer

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
        CATERVA_MAX_DIM
        CATERVA_MAX_METALAYERS

    ctypedef struct caterva_config_t:
        void *(*alloc)(size_t)
        void (*free)(void *)
        int compcodec
        int complevel
        int usedict
        int nthreads
        uint8_t filters[BLOSC2_MAX_FILTERS]
        uint8_t filtersmeta[BLOSC2_MAX_FILTERS]
        blosc2_prefilter_fn prefilter
        blosc2_prefilter_params *pparams


    ctypedef struct caterva_context_t:
        caterva_config_t *cfg


    ctypedef enum caterva_storage_backend_t:
        CATERVA_STORAGE_BLOSC
        CATERVA_STORAGE_PLAINBUFFER

    ctypedef struct caterva_metalayer_t:
        char *name
        uint8_t *sdata
        int32_t size

    ctypedef struct caterva_storage_properties_blosc_t:
        int32_t chunkshape[CATERVA_MAX_DIM]
        int32_t blockshape[CATERVA_MAX_DIM]
        bool enforceframe
        char* filename
        caterva_metalayer_t metalayers[CATERVA_MAX_METALAYERS]
        int32_t nmetalayers

    ctypedef struct caterva_storage_properties_plainbuffer_t:
        char* filename


    ctypedef union caterva_storage_properties_t:
        caterva_storage_properties_blosc_t blosc
        caterva_storage_properties_plainbuffer_t plainbuffer


    ctypedef struct caterva_storage_t:
        caterva_storage_backend_t backend
        caterva_storage_properties_t properties


    ctypedef struct caterva_params_t:
        int64_t shape[CATERVA_MAX_DIM]
        uint8_t ndim
        uint8_t itemsize


    cdef struct part_cache_s:
        uint8_t *data
        int32_t nchunk

    ctypedef struct caterva_array_t:
        caterva_storage_backend_t storage
        blosc2_schunk *sc
        uint8_t *buf
        int64_t shape[CATERVA_MAX_DIM]
        int32_t chunkshape[CATERVA_MAX_DIM]
        int64_t extshape[CATERVA_MAX_DIM]
        int32_t blockshape[CATERVA_MAX_DIM]
        int64_t extchunkshape[CATERVA_MAX_DIM]
        int32_t next_chunkshape[CATERVA_MAX_DIM]
        int64_t size
        int32_t chunksize
        int64_t extsize
        int32_t blocksize
        int64_t extchunksize
        int64_t next_chunksize
        int8_t ndim
        int8_t itemsize
        bool empty
        bool filled
        int64_t nparts
        part_cache_s part_cache


    int caterva_context_new(caterva_config_t *cfg, caterva_context_t **ctx)

    int caterva_context_free(caterva_context_t **ctx)

    int caterva_array_empty(caterva_context_t *ctx, caterva_params_t *params, caterva_storage_t *storage,
                            caterva_array_t **array)

    int caterva_array_free(caterva_context_t *ctx, caterva_array_t **array)

    int caterva_array_append(caterva_context_t *ctx, caterva_array_t *array, void *chunk, int64_t chunksize)

    int caterva_array_from_frame(caterva_context_t *ctx, blosc2_frame *frame, bool copy, caterva_array_t **array)

    int caterva_array_from_sframe(caterva_context_t *ctx, uint8_t *sframe, int64_t len, bool copy,
                                  caterva_array_t **array)

    int caterva_array_from_file(caterva_context_t *ctx, const char *filename, bool copy, caterva_array_t **array)

    int caterva_array_from_buffer(caterva_context_t *ctx, void *buffer, int64_t buffersize, caterva_params_t *params,
        caterva_storage_t *storage, caterva_array_t **array)

    int caterva_array_to_buffer(caterva_context_t *ctx, caterva_array_t *array, void *buffer, int64_t buffersize)

    int caterva_array_get_slice(caterva_context_t *ctx, caterva_array_t *src, int64_t *start, int64_t *stop,
        caterva_storage_t *storage, caterva_array_t **array)

    int caterva_array_squeeze(caterva_context_t *ctx, caterva_array_t *array)

    int caterva_array_get_slice_buffer(caterva_context_t *ctx, caterva_array_t *src, int64_t *start, int64_t *stop,
                                       int64_t *shape, void *buffer, int64_t buffersize)

    int caterva_array_set_slice_buffer(caterva_context_t *ctx, void *buffer, int64_t buffersize, int64_t *start,
                                       int64_t *stop, caterva_array_t *array)

    int caterva_array_copy(caterva_context_t *ctx, caterva_array_t *src, caterva_storage_t *storage,
                           caterva_array_t **array)


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
config_dflts = {
    'cname': 'lz4',
    'clevel': 5,
    'usedict': False,
    'nthreads': 1,
    'filters': [BLOSC_SHUFFLE],
    'filtersmeta': [0],  # no actual meta info for SHUFFLE, but anyway...
    }


cdef class Context:
    cdef caterva_context_t *context_
    cdef uint8_t compcode
    cdef uint8_t complevel
    cdef int usedict
    cdef int16_t nthreads
    cdef int32_t blocksize
    cdef uint8_t filters[BLOSC2_MAX_FILTERS]
    cdef uint8_t filtersmeta[BLOSC2_MAX_FILTERS]
    cdef blosc2_prefilter_fn prefilter
    cdef blosc2_prefilter_params* pparams

    def __init__(self, **kwargs):
        compname = kwargs.get('cname', config_dflts['cname'])
        if isinstance(compname, bytes):
            compname = compname.decode()
        if compname not in cnames2codecs:
            raise ValueError(f"'{compname}' is not among the list of available codecs ({cnames2codecs.keys()})")
        cdef caterva_config_t config
        config.free = free
        config.alloc = malloc
        config.compcodec = cnames2codecs[compname]
        config.complevel = kwargs.get('clevel', config_dflts['clevel'])
        config.usedict =  kwargs.get('usedict', config_dflts['usedict'])
        config.nthreads = kwargs.get('nthreads', config_dflts['nthreads'])
        config.prefilter = NULL
        config.pparams = NULL

        for i in range(BLOSC2_MAX_FILTERS):
            config.filters[i] = 0
            config.filtersmeta[i] = 0

        filters = kwargs.get('filters', config_dflts['filters'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters), BLOSC2_MAX_FILTERS):
            config.filters[i] = filters[i - BLOSC2_MAX_FILTERS + len(filters)]

        filtersmeta = kwargs.get('filtersmeta', config_dflts['filtersmeta'])
        for i in range(BLOSC2_MAX_FILTERS - len(filtersmeta), BLOSC2_MAX_FILTERS):
            self.filtersmeta[i] = filtersmeta[i - BLOSC2_MAX_FILTERS + len(filtersmeta)]

        caterva_context_new(&config, &self.context_)

    def __dealloc__(self):
        caterva_context_free(&self.context_)

    def tocapsule(self):
        return PyCapsule_New(self.context_, "caterva_ctx_t*", NULL)


cdef class WriteIter:
    cdef Container arr
    cdef Context ctx
    cdef buffer
    cdef dtype
    cdef buffer_shape
    cdef int32_t buffer_len
    cdef int32_t part_len

    # TODO: is np.dtype really necessary here?  Container does not have this notion, so...
    def __init__(self, Container arr):
        self.arr = arr
        self.dtype = np.dtype(f"S{arr.itemsize}")
        self.part_len = self.arr.array.chunksize * self.arr.itemsize
        self.ctx = Context()  # TODO: Use **kwargs

    def __iter__(self):
        self.buffer_shape = None
        self.buffer_len = 0
        self.buffer = None
        self.memview = None
        return self

    def __next__(self):
        cdef char* data_pointer

        if self.buffer is not None:
            data_pointer = <char*> self.buffer
            caterva_array_append(self.ctx.context_, self.arr.array, data_pointer, self.buffer_len)

        if self.arr.array.filled:
            raise StopIteration

        aux = [self.arr.array.extshape[i] // self.arr.array.chunkshape[i] for i in range(self.arr.array.ndim)]
        start_ = [0 for _ in range(self.arr.array.ndim)]
        inc = 1
        for i in range(self.arr.array.ndim - 1, -1, -1):
            start_[i] = self.arr.array.nparts % (aux[i] * inc) // inc
            start_[i] *= self.arr.array.chunkshape[i]
            inc *= aux[i]

        stop_ = [start_[i] + self.arr.array.chunkshape[i] for i in range(self.arr.array.ndim)]
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
    cdef itershape
    cdef dtype
    cdef nparts
    cdef object IterInfo

    def __init__(self, Container arr, itershape):
        if not arr.filled:
            raise ValueError("Container is not completely filled")
        self.arr = arr
        if itershape is None:
            itershape = arr.chunkshape
        self.itershape = itershape
        self.nparts = 0
        self.IterInfo = namedtuple("IterInfo", "slice, shape, size")

    def __iter__(self):
        return self

    def __next__(self):
        ndim = self.arr.ndim
        shape = tuple(self.arr.shape)
        eshape = [0 for i in range(ndim)]
        for i in range(ndim):
            if shape[i] % self.itershape[i] == 0:
                eshape[i] = self.itershape[i] * (shape[i] // self.itershape[i])
            else:
                eshape[i] = self.itershape[i] * (shape[i] // self.itershape[i] + 1)
        aux = [eshape[i] // self.itershape[i] for i in range(ndim)]
        if self.nparts >= np.prod(aux):
            raise StopIteration

        start_ = [0 for _ in range(ndim)]
        inc = 1
        for i in range(ndim - 1, -1, -1):
            start_[i] = self.nparts % (aux[i] * inc) // inc
            start_[i] *= self.itershape[i]
            inc *= aux[i]

        stop_ = [start_[i] + self.itershape[i] for i in range(ndim)]
        for i in range(ndim):
            if stop_[i] > shape[i]:
                stop_[i] = shape[i]

        sl = tuple([slice(start_[i], stop_[i]) for i in range(ndim)])
        sh = [s.stop - s.start for s in sl]
        info = self.IterInfo(slice=sl, shape=sh, size=np.prod(sh))
        self.nparts += 1

        buf = self.arr.__getitem__(info.slice)
        return buf, info



cdef get_caterva_start_stop(ndim, key, shape):
    start = tuple(s.start if s.start is not None else 0 for s in key)
    stop = tuple(s.stop if s.stop is not None else sh for s, sh in zip(key, shape))
    chunkshape = tuple(sp - st for st, sp in zip(start, stop))

    size = np.prod([stop[i] - start[i] for i in range(ndim)])

    return start, stop, chunkshape, size


cdef create_caterva_params(caterva_params_t *params, shape, itemsize):
    params.ndim = len(shape)
    params.itemsize = itemsize
    for i in range(params.ndim):
        params.shape[i] = shape[i]


cdef create_caterva_storage(caterva_storage_t *storage, kwargs):
    chunkshape = kwargs.get('chunkshape', None)
    blockshape = kwargs.get('blockshape', None)
    filename = kwargs.get('filename', None)
    if (filename is not None):
        enforceframe = True
    else:
        enforceframe = kwargs.get('enforceframe', False)
    metalayers = kwargs.get('metalayers', None)

    if chunkshape is not None and blockshape is not None:
        storage.backend = CATERVA_STORAGE_BLOSC
    else:
        storage.backend = CATERVA_STORAGE_PLAINBUFFER

    if storage.backend is CATERVA_STORAGE_BLOSC:
        if filename is not None:
            filename = filename.encode("utf-8") if isinstance(filename, str) else filename
            storage.properties.blosc.filename = filename
        else:
            storage.properties.blosc.filename = NULL
        storage.properties.blosc.enforceframe = enforceframe
        for i in range(len(chunkshape)):
            storage.properties.blosc.chunkshape[i] = chunkshape[i]
            storage.properties.blosc.blockshape[i] = blockshape[i]

        if metalayers is None:
            storage.properties.blosc.nmetalayers = 0
        else:
            storage.properties.blosc.nmetalayers = len(metalayers)
            for i, (name, content) in enumerate(metalayers.items()):
                name2 = name.encode("utf-8") if isinstance(name, str) else name # do a copy
                content = msgpack.packb(content)
                storage.properties.blosc.metalayers[i].name = strdup(name2)
                storage.properties.blosc.metalayers[i].sdata = <uint8_t *> strdup(content)
                storage.properties.blosc.metalayers[i].size = len(content)

    else:
        storage.properties.plainbuffer.filename = NULL  # Not implemented yet


cdef class Container:
    cdef caterva_array_t *array
    cdef kwargs
    cdef usermeta_len
    cdef view
    cdef sdata

    @property
    def shape(self):
        """The shape of this container."""
        return tuple([self.array.shape[i] for i in range(self.array.ndim)])

    @property
    def chunkshape(self):
        """The chunk shape of this container."""
        if self.array.storage is CATERVA_STORAGE_PLAINBUFFER:
            return None
        return tuple([self.array.chunkshape[i] for i in range(self.array.ndim)])

    @property
    def blockshape(self):
        """The block shape of this container."""
        if self.array.storage is CATERVA_STORAGE_PLAINBUFFER:
            return None
        return tuple([self.array.blockshape[i] for i in range(self.array.ndim)])

    @property
    def cratio(self):
        """The compression ratio for this container."""
        if self.array.storage is CATERVA_STORAGE_PLAINBUFFER:
            return 1
        return self.array.sc.nbytes / self.array.sc.cbytes

    @property
    def itemsize(self):
        """The itemsize of this container."""
        return self.array.itemsize

    @property
    def clevel(self):
        """The compression level for this container."""
        if self.chunkshape is None:
            return 1
        return self.array.sc.clevel

    @property
    def cname(self):
        """The compression codec name for this container."""
        if self.chunkshape is None:
            return None
        for compname, compcode in _cnames2codecs.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
            if compcode == self.array.sc.compcode:
                return compname

    @property
    def filters(self):
        """The filters list for this container."""
        if self.chunkshape is None:
            return None
        return [self.array.sc.filters[i] for i in range(BLOSC2_MAX_FILTERS)]

    @property
    def chunksize(self):
        """The chunk size (in items) for this container."""
        return self.array.chunksize

    @property
    def chunksize(self):
        """The block size (in items) for this container."""
        return self.array.blocksize

    @property
    def sframe(self):
        """The serialized frame for this container (if it exists).  *No copies* are made."""
        if self.array.sc.frame == NULL or self.array.sc.frame.fname != NULL:
            raise AttributeError("Container does not have a serialized frame."
                                 "  Use `.to_frame()` to get one.")
        cdef char *data = <char*> self.array.sc.frame.sdata
        cdef int64_t size = self.array.sc.frame.len
        cdef char[::1] mview = <char[:size:1]>data
        return mview

    @property
    def size(self):
        """The size (in items) for this container."""
        return self.array.size

    @property
    def nchunks(self):
        """The number of chunks in this container."""
        return int(self.array.extsize / self.array.chunksize)

    @property
    def ndim(self):
        """The number of dimensions of this container."""
        return self.array.ndim

    @property
    def filled(self):
        """Whether the container is completely filled or not."""
        return self.array.filled

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.usermeta_len = 0
        self.view = False
        self.sdata = False
        self.array = NULL

    def __getitem__(self, key):
        ndim = self.ndim
        start, stop, shape, size = get_caterva_start_stop(ndim, key, self.shape)
        buffersize = size * self.itemsize
        buffer = bytes(buffersize)
        cdef int64_t[CATERVA_MAX_DIM] start_, stop_, shape_

        for i in range(self.ndim):
            start_[i] = start[i]
            stop_[i] = stop[i]
            shape_[i] = shape[i]
        ctx = Context(**self.kwargs)
        caterva_array_get_slice_buffer(ctx.context_, self.array, start_, stop_, shape_, <void *> <char *> buffer, buffersize)
        return buffer

    def squeeze(self, **kwargs):
        ctx = Context(**kwargs)
        caterva_array_squeeze(ctx.context_, self.array)

    def copy(self, Container arr, **kwargs):
        ctx = Context(**kwargs)
        cdef caterva_storage_t storage_
        create_caterva_storage(&storage_, kwargs)

        cdef caterva_array_t *array_
        caterva_array_copy(ctx.context_, self.array, &storage_, &array_)
        arr.array = array_
        return arr

    def to_buffer(self, **kwargs):
        ctx = Context(**kwargs)
        buffersize = self.size * self.itemsize
        buffer = bytes(buffersize)
        caterva_array_to_buffer(ctx.context_, self.array, <void *> <char *> buffer, buffersize)
        return buffer

    def to_sframe(self, **kwargs):
        if not self.array.filled:
            raise NotImplementedError("The Container is not completely filled")
        if self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("The Container is backed by a plain buffer")
        ctx = Context(**kwargs)
        cdef char* fname
        cdef char* data
        cdef bytes sdata
        cdef blosc2_frame* frame
        if self.array.sc.frame != NULL:
            fname = self.array.sc.frame.fname
            if fname == NULL:
                return self.sframe
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

    def has_metalayer(self, name):
        if self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("Invalid backend")
        name = name.encode("utf-8") if isinstance(name, str) else name
        n = blosc2_has_metalayer(self.array.sc, name)
        return False if n < 0 else True

    def get_metalayer(self, name):
        if  self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("Invalid backend")
        name = name.encode("utf-8") if isinstance(name, str) else name
        cdef uint8_t *_content
        cdef uint32_t content_len
        n = blosc2_get_metalayer(self.array.sc, name, &_content, &content_len)
        content = <char *>_content
        content = content[:content_len]  # does a copy
        free(_content)
        return content

    def update_metalayer(self, name, content):
        if  self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("Invalid backend")
        name = name.encode("utf-8") if isinstance(name, str) else name
        content_ = self.get_metalayer(name)
        if len(msgpack.packb(content_)) != len(content):
            return ValueError("The length of the content in a metalayer cannot change.")
        n = blosc2_update_metalayer(self.array.sc, name, content, len(content))
        return n

    def update_usermeta(self, content):
        if  self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("Invalid backend")
        n = blosc2_update_usermeta(self.array.sc, content, len(content), BLOSC2_CPARAMS_DEFAULTS)
        self.usermeta_len = len(content)
        return n

    def get_usermeta(self):
        if  self.array.storage != CATERVA_STORAGE_BLOSC:
            raise NotImplementedError("Invalid backend")
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
            ctx = Context(**self.kwargs)
            if self.view:
                self.array.buf = NULL
            if self.sdata:
                self.array.sc.frame.sdata = NULL
            caterva_array_free(ctx.context_, &self.array)


def from_file(Container arr, filename, copy, **kwargs):
    ctx = Context(**kwargs)

    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    if not os.path.isfile(filename):
        raise FileNotFoundError

    cdef caterva_array_t *array_
    caterva_array_from_file(ctx.context_, filename, copy, &array_)
    arr.array = array_


def from_sframe(Container arr, sframe, copy, **kwargs):
    ctx = Context(**kwargs)

    cdef char[::1] mview
    cdef uint8_t *sframe_
    if type(sframe) is bytes:
        sframe_ = sframe
    else:
        # Try to get a memoryview from the sframe object
        mview = sframe
        sframe_ = <uint8_t*> &mview[0]
    cdef caterva_array_t *array_
    caterva_array_from_sframe(ctx.context_, sframe_, len(sframe), copy, &array_)
    if copy is False:
        arr.sdata = True
    arr.array = array_


def empty(Container arr, shape, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, kwargs.get("itemsize", 8))

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_array_empty(ctx.context_, &params_, &storage_, &array_)
    arr.array = array_


def from_buffer(Container arr, buf, shape, **kwargs):
    ctx = Context(**kwargs)

    cdef caterva_params_t params_
    create_caterva_params(&params_, shape, kwargs.get("itemsize", 8))

    cdef caterva_storage_t storage_
    create_caterva_storage(&storage_, kwargs)

    cdef caterva_array_t *array_
    caterva_array_from_buffer(ctx.context_, <void*> <char *> buf, len(buf), &params_, &storage_, &array_)
    arr.array = array_


def get_pointer(Container arr, **kwargs):
    cdef uintptr_t pointer_ = <uintptr_t> arr.array.buf
    return pointer_


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
