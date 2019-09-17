# Releasing cat4py

:Author: The Blosc Developer Team
:Contact: blosc@blosc.org
:Date: 2019-09-17


## Preliminaries

* Make sure that the current master branch is passing the tests on Microsoft Azure.

* Make sure that `RELEASE_NOTES.md` and `ANNOUNCE.md` are up to date with the latest news
in the release.

* Check any copyright listings and update them if necessary. You can use ``grep
  -i copyright`` to figure out where they might be.

* Commit the changes:

  $ git commit -a -m "Getting ready for release X.Y.Z"
  $ git push

* Check that the documentation is correctly created in https://cat4py.readthedocs.io.


## Tagging

* Create a signed tag ``X.Y.Z`` from ``master``.  Use the next message:

    $ git tag -a vX.Y.Z -m "Tagging version X.Y.Z"

* Push the tag to the github repo:

    $ git push
    $ git push --tags

After the tag would be up, update the release notes in:

## Packaging

* Make sure that you are in a clean directory.  The best way is to
  re-clone and re-build:

  $ cd /tmp
  $ git clone git@github.com:Blosc/cat4py.git
  $ cd cat4py
  $ git submodule init
  $ git submodule update
  $ CFLAGS="" python setup.py build_ext

* Check that all Cython generated ``*.c`` files are present.

* Make the tarball with the command:

  $ python setup.py sdist

Do a quick check that the tarball is sane.


## Uploading

* Register and upload it also in the PyPi repository:

    $ twine upload dist/*


## Announcing

* Send an announcement to the Blosc list.  Use the ``ANNOUNCE.md`` file as skeleton
(or possibly as the definitive version).

* Announce in Twitter via @Blosc2 account and rejoice.


## Post-release actions

* Change back to the actual cat4py repo:

  $ cd $HOME/blosc/cat4py

* Create new headers for adding new features in ``RELEASE_NOTES.md``
  add this place-holder:

  XXX version-specific blurb XXX

* Commit your changes with:

  $ git commit -a -m"Post X.Y.Z release actions done"
  $ git push


That's all folks!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
