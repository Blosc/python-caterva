#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import print_function

import os
import sys
import io

from skbuild import setup
from textwrap import dedent


if __name__ == '__main__':

    with io.open('README.md', encoding='utf-8') as f:
        long_description = f.read()

    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
    except Exception:
        # newer cpuinfo versions fail to import on unsupported architectures
        cpu_info = None

    ########### Check versions ##########
    def exit_with_error(message):
        print('ERROR: %s' % message)
        sys.exit(1)

    # Check for Python
    if sys.version_info[0] == 3:
        if sys.version_info[1] < 6:
            exit_with_error("You need Python 3.6 or greater to install Caterva!")
    else:
        exit_with_error("You need Python 3.6 or greater to install Caterva!")

    ########### End of checks ##########

    # Read the long_description from README.rst
    with open('README.md') as f:
        long_description = f.read()

    # Blosc version
    VERSION = open('VERSION').read().strip()
    # Create the version.py file
    open('caterva/version.py', 'w').write('__version__ = "%s"\n' % VERSION)

    def cmake_bool(cond):
        return 'ON' if cond else 'OFF'

    classifiers = dedent("""\
    Development Status :: 3 - Alpha
    Intended Audience :: Developers
    Intended Audience :: Information Technology
    Intended Audience :: Science/Research
    License :: OSI Approved :: BSD License
    Programming Language :: Python
    Topic :: Software Development :: Libraries :: Python Modules
    Operating System :: Microsoft :: Windows
    Operating System :: Unix
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Programming Language :: Python :: 3.8
    """)

    setup(name = "caterva",
        version = VERSION,
        description='Caterva for Python (multidimensional compressed data containers).',
        long_description = long_description,
        classifiers = [c for c in classifiers.split("\n") if c],
        author='Blosc Development Team',
        author_email='blosc@blosc.org',
        maintainer='Blosc Development Team',
        maintainer_email='blosc@blosc.org',
        url='https://github.com/Blosc/cat4py',
        license = 'https://opensource.org/licenses/BSD-3-Clause',
        platforms = ['any'],
        cmake_args = [
            '-DDEACTIVATE_SSE2:BOOL=%s' % cmake_bool(('DISABLE_BLOSC_SSE2' in os.environ) or (cpu_info is None) or ('sse2' not in cpu_info['flags'])),
            '-DDEACTIVATE_AVX2:BOOL=%s' % cmake_bool(('DISABLE_BLOSC_AVX2' in os.environ) or (cpu_info is None) or ('avx2' not in cpu_info['flags'])),
            '-DDEACTIVATE_LZ4:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_LZ4', '1'))),
            # Snappy is disabled by default
            '-DDEACTIVATE_SNAPPY:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_SNAPPY', '0'))),
            '-DDEACTIVATE_ZLIB:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZLIB', '1'))),
            '-DDEACTIVATE_ZSTD:BOOL=%s' % cmake_bool(not int(os.environ.get('INCLUDE_ZSTD', '1'))),
        ],
        install_requires=[
            'ndindex>=1.4',
            'msgpack >= 0.6.1'
        ],
        setup_requires=[
            'cython>=0.29',
            'scikit-build',
            'pytest>=3.4.2'
        ],
        tests_require=['numpy', 'psutil'],
        packages=['caterva'],
        package_dir={'caterva': 'caterva'},
        )
elif __name__ == '__mp_main__':
    # This occurs from `cpuinfo 4.0.0` using multiprocessing to interrogate the
    # CPUID flags
    # https://github.com/workhorsy/py-cpuinfo/issues/108
    pass
