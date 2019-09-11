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
        BLOSC2_MAX_FILTERS
        BLOSC2_MAX_METALAYERS
        BLOSC2_PREFILTER_INPUTS_MAX

    ctypedef struct blosc2_frame_metalayer
    ctypedef struct blosc2_frame
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
        int32_t nchunks;
        int64_t nbytes;
        int64_t cbytes;
        uint8_t** data;
        blosc2_frame* frame;
        blosc2_context* cctx;
        blosc2_context* dctx;
        uint8_t* reserved;

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

    int blosc2_has_metalayer(blosc2_schunk *schunk, char *name)
    int blosc2_add_metalayer(blosc2_schunk *schunk, char *name, uint8_t *content, uint32_t content_len)
    int blosc2_update_metalayer(blosc2_schunk *schunk, char *name, uint8_t *content, uint32_t content_len)
    int blosc2_get_metalayer(blosc2_schunk *schunk, char *name, uint8_t **content, uint32_t *content_len)

    int blosc2_update_usermeta(blosc2_schunk *schunk, uint8_t *content, int32_t content_len, blosc2_cparams cparams)
    int blosc2_get_usermeta(blosc2_schunk* schunk, uint8_t** content)


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
    caterva_array_t *caterva_from_file(caterva_ctx_t *ctx, const char *filename)
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


defaults = {'itemsize': 4,
            'compcode': 0,
            'clevel': 5,
            'use_dict': 0,
            'cnthreads': 1,
            'dnthreads': 1,
            'blocksize': 0,
            'filters': [1],
            'filters_meta': [0],
            }


cdef class CParams:
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
        self.itemsize = kargs.get('itemsize', defaults['itemsize'])
        self.compcode = kargs.get('compcode', defaults['compcode'])
        self.clevel = kargs.get('clevel', defaults['clevel'])
        self.use_dict = kargs.get('use_dict', defaults['use_dict'])
        self.nthreads = kargs.get('cnthreads', defaults['cnthreads'])
        self.blocksize = kargs.get('blocksize', defaults['blocksize'])
        self.prefilter = NULL  # TODO: implement support for prefilters
        self.pparams = NULL    # TODO: implement support for prefilters

        # Filter pipeline
        for i in range(BLOSC2_MAX_FILTERS):
            self.filters[i] = 0
        for i in range(BLOSC2_MAX_FILTERS):
            self.filters_meta[i] = 0

        filters = kargs.get('filters', defaults['filters'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters), BLOSC2_MAX_FILTERS):
            self.filters[i] = filters[i - BLOSC2_MAX_FILTERS + len(filters)]

        filters_meta = kargs.get('filters_meta', defaults['filters_meta'])
        for i in range(BLOSC2_MAX_FILTERS - len(filters_meta), BLOSC2_MAX_FILTERS):
            self.filters_meta[i] = filters_meta[i - BLOSC2_MAX_FILTERS + len(filters_meta)]


cdef class DParams:
    cdef int nthreads
    cdef void* schunk

    def __init__(self, **kargs):
        self.nthreads = kargs.get('dnthreads', defaults['dnthreads'])


cdef class Context:
    cdef caterva_ctx_t *_ctx
    cdef CParams cparams
    cdef DParams dparams

    def __init__(self, CParams cparams=None, DParams dparams=None):
        cdef blosc2_cparams _cparams
        if cparams is None:
            cparams=CParams()
        self.cparams = cparams
        _cparams.typesize = cparams.itemsize  # TODO: typesize -> itemsize in c-blosc2
        _cparams.compcode = cparams.compcode
        _cparams.clevel = cparams.clevel
        _cparams.use_dict = cparams.use_dict
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

    def __init__(self, arr):
        self.arr = arr
        dtypes = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64, 16: np.complex128}
        self.dtype = np.dtype(dtypes[arr.itemsize])

    def __iter__(self):
        self.buffer_shape = None
        self.buffer = None
        return self

    def __next__(self):
        if self.buffer is not None:
            item = np.frombuffer(self.buffer, self.dtype).reshape(self.buffer_shape)
            item = np.pad(item, [(0, self.arr.array.pshape[i] - item.shape[i]) for i in range(self.arr.ndim)], mode='constant', constant_values=0)
            item = bytes(item)
            caterva_append(self.arr.array, <char *> item, self.arr.array.psize * np.dtype(self.dtype).itemsize)

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
        self.buffer_shape = shape
        IterInfo = namedtuple("IterInfo", "slice, shape, size")
        info = IterInfo(slice=sl, shape=shape, size=np.prod(shape))

        a = bytearray(np.empty(info.shape, dtype=self.dtype))
        self.buffer = a
        return a, info


cdef class ReadIter:
    cdef Container arr
    cdef blockshape
    cdef dtype
    cdef nparts

    def __init__(self, arr, blockshape):
        if not arr.filled:
            print("Container is not filled")
            raise AttributeError
        self.arr = arr
        self.blockshape = blockshape
        self.nparts = 0

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
        IterInfo = namedtuple("IterInfo", "slice, shape, size")
        info = IterInfo(slice=sl, shape=sh, size=np.prod(sh))
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

    @property
    def shape(self):
        cdef caterva_dims_t shape = caterva_get_shape(self.array)
        return tuple([shape.dims[i] for i in range(shape.ndim)])

    @property
    def pshape(self):
        if self.array.storage == CATERVA_STORAGE_PLAINBUFFER:
            return None
        cdef caterva_dims_t pshape = caterva_get_pshape(self.array)
        return tuple([pshape.dims[i] for i in range(pshape.ndim)])

    @property
    def cratio(self):
        if self.array.storage is not CATERVA_STORAGE_BLOSC:
            return 1
        return self.array.sc.nbytes / self.array.sc.cbytes

    @property
    def itemsize(self):
        return self.array.ctx.cparams.typesize

    @property
    def clevel(self):
        return self.array.ctx.cparams.clevel

    @property
    def compcode(self):
        return self.array.ctx.cparams.compcode

    @property
    def filters(self):
        return [self.array.ctx.cparams.filters[i] for i in range(BLOSC2_MAX_FILTERS)]

    @property
    def size(self):
        return self.array.size

    @property
    def psize(self):
        return self.array.psize

    @property
    def npart(self):
        return int(self.array.esize / self.array.psize)

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def filled(self):
        return self.array.filled

    def __init__(self, **kwargs):
        cparams = CParams(**kwargs)
        dparams = DParams(**kwargs)
        self.ctx = Context(cparams, dparams)
        cdef caterva_ctx_t * ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(self.ctx.tocapsule(), "caterva_ctx_t*")
        self.usermeta_len = 0

        cdef int64_t *pshape_
        cdef caterva_dims_t _pshape
        cdef blosc2_frame *_frame

        pshape = kwargs['pshape'] if 'pshape' in kwargs else None
        filename = kwargs['filename'] if 'filename' in kwargs else None
        if pshape is None:
            if filename is not None:
                raise NotImplementedError
            else:
                self.array = caterva_empty_array(ctx_, NULL, NULL)
        else:
            ndim = len(pshape)

            pshape_ = <int64_t*> malloc(ndim * sizeof(int64_t))
            for i in range(ndim):
                pshape_[i] = pshape[i]
            _pshape = caterva_new_dims(pshape_, ndim)
            free(pshape_)

            if filename is None:
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

    def __dealloc__(self):
        if self.array != NULL:
            caterva_free_array(self.array)


def from_file(Container arr, filename):
    ctx = Context()
    cdef caterva_ctx_t * ctx_ = <caterva_ctx_t*> PyCapsule_GetPointer(ctx.tocapsule(), "caterva_ctx_t*")
    filename = filename.encode("utf-8") if isinstance(filename, str) else filename
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef caterva_array_t *a_ = caterva_from_file(ctx_, filename)
    arr.ctx = ctx
    arr.array = a_


def getitem(Container cont, key):
    cdef caterva_dims_t _start, _stop, _pshape
    ndim = cont.array.ndim
    _start, _stop, _pshape, size = get_caterva_start_stop(ndim, key, cont.shape)
    bsize = size * cont.itemsize
    buffer = bytes(bsize)
    err = caterva_get_slice_buffer(<char *> buffer, cont.array, &_start, &_stop, &_pshape)
    return buffer


def setitem(Container cont, key, item):
    if not cont.array.filled or cont.array.storage == CATERVA_STORAGE_BLOSC:
        raise NotImplementedError
    cdef caterva_dims_t _start, _stop, _pshape
    ndim = cont.array.ndim
    _start, _stop, _pshape, size = get_caterva_start_stop(ndim, key, cont.shape)
    caterva_set_slice_buffer(cont.array, <void *> <char *> item, &_start, &_stop)


def copy(Container src, Container dest):
    caterva_copy(dest.array, src.array)


def to_buffer(Container arr):
    cdef caterva_dims_t shape_ = caterva_get_shape(arr.array)
    shape = []
    for i in range(shape_.ndim):
        shape.append(shape_.dims[i])
    size = np.prod(shape) * arr.array.ctx.cparams.typesize

    buffer = bytes(size)
    caterva_to_buffer(arr.array, <void *> <char *> buffer)
    return buffer


def from_buffer(Container arr, shape, buf):
    if arr.pshape is not None:
        assert(len(shape) == len(arr.pshape))

    cdef caterva_dims_t _shape = get_caterva_shape(shape)
    cdef int retcode = caterva_from_buffer(arr.array, &_shape, <void*> <char *> buf)
    if retcode < 0:
        raise ValueError("Error filling the caterva object with buffer")


def has_metalayer(Container arr, name):
    if  arr.array.storage != CATERVA_STORAGE_BLOSC and arr.array.sc.frame == NULL:
        return NotImplementedError

    name = name.encode("utf-8") if isinstance(name, str) else name
    n = blosc2_has_metalayer(arr.array.sc, name)
    return False if n < 0 else True


def get_metalayer(Container arr, name):
    if  arr.array.storage != CATERVA_STORAGE_BLOSC and arr.array.sc.frame == NULL:
        return NotImplementedError

    name = name.encode("utf-8") if isinstance(name, str) else name
    cdef uint8_t *_content
    cdef uint32_t content_len
    n = blosc2_get_metalayer(arr.array.sc, name, &_content, &content_len)
    content = <char *>_content
    content = content[:content_len]  # does a copy
    free(_content)
    return content


def update_metalayer(Container arr, name, content):
     name = name.encode("utf-8") if isinstance(name, str) else name
     content_ = get_metalayer(arr, name)
     if len(content_) != len(content):
         return ValueError("The length of the content in a metalayer cannot change.")
     n = blosc2_update_metalayer(arr.array.sc, name, content, len(content))
     return n


def update_usermeta(Container arr, content):
    n = blosc2_update_usermeta(arr.array.sc, content, len(content), BLOSC2_CPARAMS_DEFAULTS)
    arr.usermeta_len = len(content)
    return n


def get_usermeta(Container arr):
    cdef uint8_t *_content
    n = blosc2_get_usermeta(arr.array.sc, &_content)
    if n < 0:
        raise ValueError("Cannot get the usermeta section")
    content = <char *>_content
    content = content[:n]  # does a copy
    free(_content)
    return content
