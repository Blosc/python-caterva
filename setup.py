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
if any([(3,) < v < (3, 5)]):
    raise Exception(
        "Unsupported Python version %d.%d. Requires Python >= 3.5 " % v[:2])

import os
from glob import glob
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from pkg_resources import resource_filename

# For guessing the capabilities of the CPU for C-Blosc2
try:
    # Currently just Intel and some ARM archs are supported by cpuinfo module
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
except:
    cpu_info = {'flags': []}


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """
    def __contains__(self, key):
        return (
            key == 'build_ext'
            or super(LazyCommandClass, self).__contains__(key)
        )

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super(LazyCommandClass, self).__setitem__(key, value)

    def __getitem__(self, key):
        if key != 'build_ext':
            return super(LazyCommandClass, self).__getitem__(key)

        from Cython.Distutils import build_ext as cython_build_ext

        class build_ext(cython_build_ext):
            """
            Custom build_ext command that lazily adds numpy's include_dir to
            extensions.
            """
            def build_extensions(self):
                """
                Lazily append numpy's include directory to Extension includes.

                This is done here rather than at module scope because setup.py
                may be run before numpy has been installed, in which case
                importing numpy and calling `numpy.get_include()` will fail.
                """
                numpy_incl = resource_filename('numpy', 'core/include')
                for ext in self.extensions:
                    ext.include_dirs.append(numpy_incl)

                super(cython_build_ext, self).build_extensions()
        return build_ext


# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc2 dir if installed in the system
BLOSC2_DIR = os.environ.get('BLOSC2_DIR', '')

# Sources & libraries
inc_dirs = ['cat4py']
lib_dirs = []
libs = []
def_macros = []
sources = ['cat4py/container_ext.pyx']

optional_libs = []

# Handle --blosc2=[PATH] --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
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
    # Using the Blosc library
    lib_dirs += [os.path.join(BLOSC2_DIR, 'lib')]
    inc_dirs += [os.path.join(BLOSC2_DIR, 'include')]
    libs += ['blosc']
else:
    # Compiling everything from sources
    sources += [f for f in glob('c-blosc2/blosc/*.c')
                if 'avx2' not in f and 'sse2' not in f and 'neon' not in f]
    sources += glob('c-blosc2/internal-complibs/lz4*/*.c')
    sources += glob('c-blosc2/internal-complibs/snappy*/*.cc')
    sources += glob('c-blosc2/internal-complibs/zlib*/*.c')
    sources += glob('c-blosc2/internal-complibs/zstd*/*/*.c')
    inc_dirs += [os.path.join('c-blosc2', 'blosc')]
    inc_dirs += [d for d in glob('c-blosc2/internal-complibs/*')
                 if os.path.isdir(d)]
    inc_dirs += [d for d in glob('c-blosc2/internal-complibs/zstd*/*')
                 if os.path.isdir(d)]
    def_macros += [('HAVE_LZ4', 1), ('HAVE_ZLIB', 1), ('HAVE_ZSTD', 1)]

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
        'Programming Language :: Python :: 3.5',
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
        'setuptools_scm>=3.2.0'
    ],
    tests_require=tests_require,
    extras_require=dict(
        optional=[
        ],
        test=tests_require
    ),
    packages=find_packages(),
    package_data={'cat4py': ['container_ext.pxd']},
    cmdclass=LazyCommandClass(),
)
