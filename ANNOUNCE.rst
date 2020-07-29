# Announcing cat4py 0.3.0


## What is new?

In this realease the development status has changed to alpha and the instructions about
installing cat4py from pip has been added.

A new class, named `NPArray`, has been created for better integration with numpy arrays.

Another functionality to consider is that now the containers can be serialized/deserialized
to/from frames (bytes).


For more info, you can have a look at the release notes in:

https://github.com/Blosc/cat4py/blob/master/RELEASE_NOTES.md

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
