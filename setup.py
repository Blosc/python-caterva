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


with io.open('README.md', encoding='utf-8') as f:
    long_description = f.read()


def exit_with_error(message):
    print('ERROR: %s' % message)
    sys.exit(1)


# Check for Python
if sys.version_info[0] == 3:
    if sys.version_info[1] < 6:
        exit_with_error("You need Python 3.6 or greater to install Caterva!")
else:
    exit_with_error("You need Python 3.6 or greater to install Caterva!")


# Read the long_description from README.md
with open('README.md') as f:
    long_description = f.read()

# Blosc version
VERSION = open('VERSION').read().strip()
# Create the version.py file
open('caterva/version.py', 'w').write('__version__ = "%s"\n' % VERSION)


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


setup(
    name="caterva",
    version=VERSION,
    description='Caterva for Python (multidimensional compressed data containers).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[c for c in classifiers.split("\n") if c],
    author='Blosc Development Team',
    author_email='blosc@blosc.org',
    maintainer='Blosc Development Team',
    maintainer_email='blosc@blosc.org',
    url='https://github.com/Blosc/cat4py',
    license='https://opensource.org/licenses/BSD-3-Clause',
    platforms=['any'],
    packages=['caterva'],
    package_dir={'caterva': 'caterva'},
    install_requires=['ndindex'],
)
