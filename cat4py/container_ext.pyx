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
        int64_t extendedshape[CATERVA_MAX_DIM]
        int64_t size
        int32_t chunksize
        int64_t extendedesize
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
    'compname': 'lz4',
    'complevel': 5,
    'usedict': False,
    'nthreads': 1,
    'blocksize': 0,
    'filters': [BLOSC_SHUFFLE],
    'filtersmeta': [0],  # no actual meta info for SHUFFLE, but anyway...
    }


cdef class Context:
    cdef caterva_context_t *_context
    cdef str compname
    cdef uint8_t compcode
    cdef uint8_t compclevel
    cdef int usedict
    cdef int16_t nthreads
    cdef int32_t blocksize
    cdef uint8_t filters[BLOSC2_MAX_FILTERS]
    cdef uint8_t filtersmeta[BLOSC2_MAX_FILTERS]
    cdef blosc2_prefilter_fn prefilter
    cdef blosc2_prefilter_params* pparams

    def __init__(self, **kargs):
        compname = kargs.get('compname', config_dflts['compname'])
        if isinstance(compname, bytes):
            compname = compname.decode()
        if compname not in cnames2codecs:
            raise ValueError(f"'{compname}' is not among the list of available codecs ({cnames2codecs.keys()})")
        self.compname = compname
        self.compcode = cnames2codecs[compname]
        self.complevel = kargs.get('complevel', config_dflts['complevel'])
        self.usedict = kargs.get('usedict', config_dflts['usedict'])
        self.nthreads = kargs.get('nthreads', config_dflts['nthreads'])
        self.blocksize = kargs.get('blocksize', config_dflts['blocksize'])
        self.prefilter = NULL  # TODO: implement support for prefilters
        self.pparams = NULL    # TODO: implement support for prefilters

        # Filter pipeline
        for i in range(BLOSC2_MAX_FILTERS):
            self.filters[i] = 0
            self.filtersmeta[i] = 0

        filters = kargs.get('filters', config_dflts['filters'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters), BLOSC2_MAX_FILTERS):
            self.filters[i] = filters[i - BLOSC2_MAX_FILTERS + len(filters)]

        filtersmeta = kargs.get('filtersmeta', config_dflts['filtersmeta'])
        for i in range(BLOSC2_MAX_FILTERS - len(filtersmeta), BLOSC2_MAX_FILTERS):
            self.filtersmeta[i] = filtersmeta[i - BLOSC2_MAX_FILTERS + len(filtersmeta)]

        cdef caterva_config_t config
        config.free = free
        config.alloc = malloc
        config.compcodec = self.compcode
        config.complevel = self.complevel
        config.usedict = self.usedict
        config.nthreads = self.nthreads
        config.prefilter = self.prefilter
        config.pparams = self.pparams

        for i in range(BLOSC2_MAX_FILTERS):
            config.filters[i] = self.filters[i]
            config.filtersmeta[i] = self.filters_meta[i]

        caterva_context_new(&cfg, &self._context)


    def __dealloc__(self):
        caterva_context_free(&self._context)


    def tocapsule(self):
        return PyCapsule_New(self._context, "caterva_ctx_t*", NULL)


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
            raise ValueError("Container is not completely filled")
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

        buf = self.arr._slicebuffer(info.slice)
        return buf, info



cdef get_caterva_start_stop(ndim, key, shape):
    start = tuple(s.start if s.start is not None else 0 for s in key)
    stop = tuple(s.stop if s.stop is not None else sh for s, sh in zip(key, shape))
    chunkshape = tuple(sp - st for st, sp in zip(start, stop))

    size = np.prod([stop[i] - start[i] for i in range(ndim)])

    return start, stop, chunkshape, size

cdef create_caterva_storage(kwargs, caterva_storage_t *storage):
    itemsize = kwargs['itemsize'] if 'chunkshape' in kwargs else None
    chunkshape = kwargs['chunkshape'] if 'chunkshape' in kwargs else None
    filename = kwargs['filename'] if 'filename' in kwargs else None
    enforceframe = kwargs['enforceframe'] if 'enforceframe' in kwargs else False
    metalayers = kwargs['metalayers'] if 'metalayers' in kwargs else False

    if filename is True and enforceframe is False:
        raise ValueError("You cannot specify a `filename` and set `enforceframe` to False at once.")

    storage.backend = CATERVA_STORAGE_PLAINBUFFER if chunkshape is None else CATERVA_STORAGE_BLOSC
    if storage.backend is CATERVA_STORAGE_BLOSC:
        storage.properties.blosc.filename = filename
        storage.properties.blosc.enforceframe = enforceframe
        for i in range(len(chunkshape)):
            storage.properties.blosc.chunkshape[i] = chunkshape[i]
    else:
        storage.properties.plainbuffer.filename = NULL  # Not implemented yet

    if metalayers:
        metalayers = kwargs['metalayers']
        storage.properties.blosc.nmetalayers = len(metalayers)
        for i, name, content in enumerate(metalayers.items()):
            name = name.encode("utf-8") if isinstance(name, str) else name
            content = msgpack.packb(content)
            storage.properties.blosc.metalayers[i].name = name
            storage.properties.blosc.metalayers[i].sdata = content
            storage.properties.blosc.metalayers[i].size = len(content)


cdef class Container:
    cdef caterva_array_t *array
    cdef kwargs
    cdef usermeta_len

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
        return self.array.sc.clevel

    @property
    def compname(self):
        """The compression codec name for this container."""
        for compname, compcode in _cnames2codecs.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
            if compcode == self.array.sc.compcode:
                return compname

    @property
    def filters(self):
        """The filters list for this container."""
        return [self.array.sc.filters[i] for i in range(BLOSC2_MAX_FILTERS)]

    @property
    def chunksize(self):
        """The chunk size (in items) for this container."""
        return self.array.chunksize

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
        return int(self.array.extendedesize / self.array.chunksize)

    @property
    def ndim(self):
        """The number of dimensions of this container."""
        return self.array.ndim

    @property
    def filled(self):
        """Whether the container is completely filled or not."""
        return self.array.filled

    def __init__(self, shape, **kwargs):
        ctx = Context(**kwargs)
        self.kwargs = kwargs
        self.usermeta_len = 0

        ndim = len(shape)
        itemsize = kwargs['itemsize'] if 'itemsize' in kwargs else 8

        cdef caterva_params_t params
        params.itemsize = itemsize
        params.ndim = ndim
        for i in range(ndim):
            params.shape[i] = shape[i]

        cdef caterva_storage_t storage;
        create_caterva_storage(kwargs, &storage)

        caterva_array_empty(ctx._context, &params, &storage, &self.array)


    def __getitem__(self, key):
        ndim = self.ndim
        start, stop, shape, size = get_caterva_start_stop(ndim, key, self.shape)
        buffersize = size * self.itemsize
        buffer = bytes(buffersize)
        cdef int64_t[self.ndim] start_, stop_, shape_
        for i in range(self.ndim):
            start_[i], stop_[i], shape_[i] = start[i], stop[i], shape[i]

        ctx = Context(self.kwargs)
        caterva_array_get_slice_buffer(ctx._context, self.array, start_, stop_, shape_, <char *> buffer, buffersize)

        return buffer

    def squeeze(self, **kwargs):
        ctx = Context(**kwargs)
        caterva_array_squeeze(ctx._context, self.array)

    def to_buffer(self, **kwargs):
        ctx = Context(**kwargs)
        buffersize = self.size * self.itemsize
        buffer = bytes(buffersize)
        caterva_array_to_buffer(ctx._context, self.array, <void *> <char *> buffer, buffersize)
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

    def copy(self, **kargs):
        ctx = Context(**kwargs)
        cdef caterva_storage_t storage
        create_caterva_storage(kwargs, &storage)

        caterva_array_copy(ctx._context, self.array, &storage, )

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


def empty(Container arr, filename, copy):

def from_file(Container arr, filename, copy):
    ctx = Context()
    cdef caterva_ctx_t * ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(ctx.tocapsule(), "caterva_ctx_t*")
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef caterva_array_t *a_ = caterva_from_file(ctx_, filename, copy)
    arr.ctx = ctx
    arr.array = a_


def from_sframe(Container arr, sframe, copy):
    ctx = Context()
    cdef caterva_ctx_t *ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(ctx.tocapsule(), "caterva_ctx_t*")
    cdef char[::1] mview
    cdef uint8_t *frame_
    if type(sframe) is bytes:
        frame_ = sframe
    else:
        # Try to get a memoryview from the sframe object
        mview = sframe
        frame_ = <uint8_t*>&mview[0]
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
