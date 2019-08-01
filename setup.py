#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import absolute_import
from sys import version_info as v
import os
# from setuptools import find_packages
from skbuild import setup

# For guessing the capabilities of the CPU for C-Blosc2
try:
    # Currently just Intel and some ARM archs are supported by cpuinfo module
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
except ImportError:
    cpuinfo = None
    cpu_info = {'flags': []}

# Check whether this Python version is supported
if any([(3,) < v < (3, 6)]):
    raise Exception(
        "Unsupported Python version %d.%d. Requires Python >= 3.6 " % v[:2])

# Global variables
CFLAGS = os.environ.get('CFLAGS', '').split()
print("CFLAGS->", CFLAGS)
LFLAGS = os.environ.get('LFLAGS', '').split()
# Allow setting the Blosc2 dir if installed in the system
BLOSC2_DIR = os.environ.get('BLOSC2_DIR', '')
print("BLOSC2_DIR: '%s'" % BLOSC2_DIR)

# compile and link code instrumented for coverage analysis
if os.getenv('TRAVIS') and os.getenv('CI') and v[0:2] == (3, 7):
    CFLAGS.extend(["-fprofile-arcs", "-ftest-coverage"])
    LFLAGS.append("-lgcov")

print("CFLAGS2->", CFLAGS)


def cmake_bool(cond):
    return 'ON' if cond else 'OFF'


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
    install_requires=['numpy>=1.16'],
    cmake_args=[
        '-DBLOSC_DIR:PATH=%s' % os.environ.get('BLOSC_DIR', ''),
        '-DDEACTIVATE_SSE2:BOOL=%s' % cmake_bool('DISABLE_BLOSC_SSE2' in os.environ),
        '-DDEACTIVATE_AVX2:BOOL=%s' % cmake_bool('DISABLE_BLOSC_AVX2' in os.environ),
        '-DDEACTIVATE_LZ4:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_LZ4', '1'))),
        # Snappy is disabled by default
        '-DDEACTIVATE_SNAPPY:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_SNAPPY', '0'))),
        '-DDEACTIVATE_ZLIB:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZLIB', '1'))),
        '-DDEACTIVATE_ZSTD:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZSTD', '1'))),
    ],
    # packages=find_packages(),
    package_data={'cat4py': ['container_ext.pyx']},
)
