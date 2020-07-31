# Announcing cat4py 0.4.2


## What is new?

In this release the package is compiled using the scikit-build tool for a better integration
with the C dependencies.

A complete API renaming has been introduced to facilitate the use of cat4py by the community.

Finally, the buffer protocol and the array interface have been implemented.

For more info, you can have a look at the release notes in:

https://github.com/Blosc/cat4py/releases

More docs and examples are available in the documentation site:

https://cat4py.readthedocs.io


## What is it?

Caterva is an open source C library and a format that allows to store large
multidimensional, chunked, compressed datasets. Data can be stored either
in-memory or on-disk, but the API to handle both versions is the same.
Compression is handled transparently for the user by adopting the Blosc2 library.

cat4py is a pythonic wrapper for the Caterva library.


## Sources repository

The sources and documentation are managed through github services at:

http://github.com/Blosc/cat4py

Caterva is distributed using the BSD license, see
[LICENSE](https://github.com/Blosc/cat4py/blob/master/LICENSE) for details.


## Mailing list

There is an official Blosc mailing list where discussions about Caterva are welcome:

blosc@googlegroups.com

http://groups.google.es/group/blosc


Enjoy Data!
- The Blosc Development Team
