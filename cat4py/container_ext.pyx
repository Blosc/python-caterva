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
import cython
from libcpp cimport bool
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer


cdef extern from "<stdint.h>":
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long long uint64_t


cdef extern from "blosc.h":
    ctypedef enum:
        BLOSC_MAX_FILTERS
        BLOSC2_MAX_METALAYERS

    ctypedef struct blosc2_frame_metalayer

    ctypedef struct blosc2_frame

    ctypedef struct blosc2_context

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
        uint8_t filters[BLOSC_MAX_FILTERS]
        uint8_t filters_meta[BLOSC_MAX_FILTERS]
        int32_t nchunks;
        int64_t nbytes;
        int64_t cbytes;
        uint8_t* metadata_chunk
        uint8_t* userdata_chunk
        uint8_t** data
        blosc2_frame* frame
        blosc2_context* cctx
        blosc2_context* dctx
        uint8_t* reserved

    ctypedef struct blosc2_cparams:
        uint8_t compcode
        uint8_t clevel
        int use_dict
        int32_t typesize
        int16_t nthreads
        int32_t blocksize
        void* schunk
        uint8_t filters[BLOSC_MAX_FILTERS]
        uint8_t filters_meta[BLOSC_MAX_FILTERS]

    blosc2_cparams BLOSC_CPARAMS_DEFAULTS

    ctypedef struct blosc2_dparams:
        int nthreads
        void* schunk

    blosc2_dparams BLOSC_DPARAMS_DEFAULTS


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

    ctypedef struct caterva_array_t:
        caterva_ctx_t *ctx
        caterva_storage_t storage
        blosc2_schunk *sc
        uint8_t *buf
        int64_t shape[CATERVA_MAXDIM]
        int64_t pshape[CATERVA_MAXDIM]
        int64_t eshape[CATERVA_MAXDIM]
        int64_t size
        int64_t psize
        int64_t esize
        int8_t ndim

    caterva_ctx_t *caterva_new_ctx(void *(*all)(size_t), void (*free)(void *),
                                   blosc2_cparams cparams, blosc2_dparams dparams)
    int caterva_free_ctx(caterva_ctx_t *ctx)
    caterva_dims_t caterva_new_dims(int64_t *dims, int8_t ndim)
    caterva_array_t *caterva_empty_array(caterva_ctx_t *ctx, blosc2_frame *fr, caterva_dims_t *pshape)
    int caterva_free_array(caterva_array_t *carr)
    caterva_array_t *caterva_from_file(caterva_ctx_t *ctx, const char *filename)
    int caterva_from_buffer(caterva_array_t *dest, caterva_dims_t *shape, void *src)
    int caterva_fill(caterva_array_t *dest, caterva_dims_t *shape, void *value)
    int caterva_to_buffer(caterva_array_t *src, void *dest)
    int caterva_get_slice(caterva_array_t *dest, caterva_array_t *src, caterva_dims_t *start, caterva_dims_t *stop)
    int caterva_repart(caterva_array_t *dest, caterva_array_t *src)
    int caterva_squeeze(caterva_array_t *src)
    int caterva_get_slice_buffer(void *dest, caterva_array_t *src, caterva_dims_t *start,
                                 caterva_dims_t *stop, caterva_dims_t *d_pshape)
    int caterva_get_slice_buffer_no_copy(void **dest, caterva_array_t *src, caterva_dims_t *start,
                                         caterva_dims_t *stop, caterva_dims_t *d_pshape)
    int caterva_set_slice_buffer(caterva_array_t *dest, void *src, caterva_dims_t *start, caterva_dims_t *stop)
    int caterva_update_shape(caterva_array_t *src, caterva_dims_t *shape)
    caterva_dims_t caterva_get_shape(caterva_array_t *src)
    caterva_dims_t caterva_get_pshape(caterva_array_t *src)


cdef class CParams:
    cdef uint8_t compcode
    cdef uint8_t clevel
    cdef int use_dict
    cdef int32_t itemsize
    cdef int16_t nthreads
    cdef int32_t blocksize
    cdef void* schunk
    cdef uint8_t filters[BLOSC_MAX_FILTERS]
    cdef uint8_t filters_meta[BLOSC_MAX_FILTERS]

    def __cinit__(self, itemsize, compcode=0, clevel=5, use_dict=0, nthreads=1, blocksize=0, filter=1):
        self.itemsize = itemsize
        self.compcode = compcode
        self.clevel = clevel
        self.use_dict = use_dict
        self.nthreads = nthreads
        self.blocksize = blocksize
        self.filters[BLOSC_MAX_FILTERS - 1] = filter


cdef class DParams:
    cdef int nthreads
    cdef void* schunk

    def __cinit__(self, nthreads):
        self.nthreads = nthreads


cdef class Context:
    cdef caterva_ctx_t *_ctx

    def __cinit__(self, CParams cparams, DParams dparams):
        cdef blosc2_cparams _cparams
        _cparams.typesize = cparams.itemsize  # TODO: typesize -> itemsize in c-blosc2
        _cparams.compcode = cparams.compcode
        _cparams.clevel = cparams.clevel
        _cparams.use_dict = cparams.use_dict
        _cparams.nthreads = cparams.nthreads
        _cparams.blocksize = cparams.blocksize
        for i in range(BLOSC_MAX_FILTERS):
            _cparams.filters[i] = cparams.filters[i]
        cdef blosc2_dparams _dparams
        _dparams.nthreads = dparams.nthreads
        self._ctx = caterva_new_ctx(NULL, NULL, _cparams, _dparams)

    def __dealloc__(self):
        caterva_free_ctx(self._ctx)

    def to_capsule(self):
        return PyCapsule_New(self._ctx, "caterva_ctx_t*", NULL)

    def __str__(self):
        return "Caterva context object"

