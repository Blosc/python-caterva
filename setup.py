#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import absolute_import
from sys import version_info as v

# Check this Python version is supported
if any([(3,) < v < (3, 6)]):
    raise Exception(
        "Unsupported Python version %d.%d. Requires Python >= 3.6 " % v[:2])

import os
from glob import glob
import sys
import numpy

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

# For guessing the capabilities of the CPU for C-Blosc2
try:
    # Currently just Intel and some ARM archs are supported by cpuinfo module
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
except:
    cpu_info = {'flags': []}


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow looking for the Blosc2 libs and headers if installed in the system
BLOSC2_DIR = os.environ.get('BLOSC2_DIR', '')
# Allow looking for the Blosc2 libs and headers if installed in the system
CATERVA_DIR = os.environ.get('CATERVA_DIR', '')

# Sources & libraries
inc_dirs = [numpy.get_include()]
lib_dirs = []
libs = []
def_macros = []
sources = ['cat4py/container_ext.pyx']

optional_libs = []

# Handle --caterva=[PATH] --blosc2=[PATH] --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--caterva=') == 0:
        CATERVA_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--blosc2=') == 0:
        BLOSC2_DIR = os.path.expanduser(arg.split('=')[1])
        sys.argv.remove(arg)
    if arg.find('--lflags=') == 0:
        LFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)
    if arg.find('--cflags=') == 0:
        CFLAGS = arg.split('=')[1].split()
        sys.argv.remove(arg)

if BLOSC2_DIR != '':
    print(f"Using the Blosc2 library installed in {BLOSC2_DIR}")
    lib_dirs += [os.path.join(BLOSC2_DIR, 'lib')]
    inc_dirs += [os.path.join(BLOSC2_DIR, 'include')]
    libs += ['blosc2']
else:
    print(f"Compiling the included Blosc2 sources")
    sources += [f for f in glob('c-blosc2/blosc/*.c')
                if 'avx2' not in f and 'sse2' not in f and
                   'neon' not in f and 'altivec' not in f]
    sources += glob('c-blosc2/internal-complibs/lz4*/*.c')
    #sources += glob('c-blosc2/internal-complibs/miniz*/*.c')
    #sources += glob('c-blosc2/internal-complibs/zstd*/*/*.c')  # TODO: add a flag for including zstd
    # sources += glob('c-blosc2/internal-complibs/lizard*/*/*.c')
    inc_dirs += [os.path.join('c-blosc2', 'blosc')]
    inc_dirs += [d for d in glob('c-blosc2/internal-complibs/*')
                 if os.path.isdir(d)]
    # inc_dirs += [d for d in glob('c-blosc2/internal-complibs/zstd*/*')
    #              if os.path.isdir(d)]
    # TODO: when including miniz, we get a `_compress2` symbol not found error
    # def_macros += [('HAVE_LZ4', 1), ('HAVE_ZLIB', 1), ('HAVE_ZSTD', 1)]
    # def_macros += [('HAVE_LZ4', 1), ('HAVE_ZSTD', 1)]
    def_macros += [('HAVE_LZ4', 1)]
    #               ('HAVE_ZSTD', 1)  # TODO: add a flag for including zstd
    #               ('HAVE_LIZARD', 1)  # TODO: xxhash collide between ztsd and lizard

    # Guess SSE2 or AVX2 capabilities
    # SSE2
    if 'DISABLE_CAT4PY_SSE2' not in os.environ and 'sse2' in cpu_info['flags']:
        print('SSE2 detected')
        CFLAGS.append('-DSHUFFLE_SSE2_ENABLED')
        sources += [f for f in glob('c-blosc2/blosc/*.c') if 'sse2' in f]
        if os.name == 'posix':
            CFLAGS.append('-msse2')
        elif os.name == 'nt':
            def_macros += [('__SSE2__', 1)]

    # AVX2
    if 'DISABLE_CAT4PY_AVX2' not in os.environ and 'avx2' in cpu_info['flags']:
        print('AVX2 detected')
        CFLAGS.append('-DSHUFFLE_AVX2_ENABLED')
        sources += [f for f in glob('c-blosc2/blosc/*.c') if 'avx2' in f]
        if os.name == 'posix':
            CFLAGS.append('-mavx2')
        elif os.name == 'nt':
            def_macros += [('__AVX2__', 1)]

# Add Caterva sources
if CATERVA_DIR != '':
    print(f"Using the Caterva library installed in {CATERVA_DIR}")
    lib_dirs += [os.path.join(CATERVA_DIR, 'lib')]
    inc_dirs += [os.path.join(CATERVA_DIR, 'include')]
    libs += ['caterva']
else:
    print(f"Compiling the included Caterva sources")
    sources += [f for f in glob('Caterva/caterva/*.c')]
    inc_dirs += [os.path.join('Caterva', 'caterva')]

tests_require = []

# compile and link code instrumented for coverage analysis
if os.getenv('TRAVIS') and os.getenv('CI') and v[0:2] == (3, 7):
    CFLAGS.extend(["-fprofile-arcs", "-ftest-coverage"])
    LFLAGS.append("-lgcov")


setup(
    name="cat4py",
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'cat4py/version.py'
    },
    description='Caterva for Python (multidimensional compressed data containers).',
    long_description="""\

cat4py is a wrapper for the Caterva library, and provides multidimensional
and compressed data containers.  cat4py allows for storing data in both
memory and disk, allowing to work on both media exactly in the same way.
Data in these containers can be retrieved in any multidimensional slice. 
By supporting extremely fast compression by default (via the C-Blosc2
library), cat4py objects reduce memory/disk I/O needs, usually
increasing the I/O speed not only to disk, but potentially to memory too.

""",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    author='Blosc Development Team',
    author_email='blosc@blosc.org',
    maintainer='Blosc Development Team',
    maintainer_email='blosc@blosc.org',
    url='https://github.com/Blosc/cat4py',
    license='BSD',
    platforms=['any'],
    ext_modules=cythonize([
        Extension(
            'cat4py.container_ext',
            include_dirs=inc_dirs,
            define_macros=def_macros,
            sources=sources,
            library_dirs=lib_dirs,
            libraries=libs,
            extra_link_args=LFLAGS,
            extra_compile_args=CFLAGS
        )
    ]),
    install_requires=['numpy>=1.16'],
    setup_requires=[
        'cython>=0.29',
        'numpy>=1.16',
        'setuptools>=40.0',
        'setuptools_scm>=3.2.0',
        'pytest>=3.4.2',
        'msgpack>=0.6.1'
    ],
    tests_require=tests_require,
    extras_require=dict(
        optional=[
        ],
        test=tests_require
    ),
    packages=find_packages(),
    package_data={'cat4py': ['container_ext.pyx']},
)
