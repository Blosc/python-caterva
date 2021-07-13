Python-caterva release procedure
================================

Preliminaries
-------------

* Make sure that the current master branch is passing the tests on Microsoft Azure.

* Make sure that `RELEASE_NOTES.rst` and `ANNOUNCE.rst` are up to date with the latest news
  in the release.

* Check that `VERSION` file contains the correct number.

* Check any copyright listings and update them if necessary. You can use ``grep
  -i copyright`` to figure out where they might be.

* Commit the changes::

    git commit -a -m "Getting ready for release X.Y.Z"
    git push

* Check that the documentation is correctly created in https://python-caterva.readthedocs.io.


Tagging
-------

* Create a signed tag ``X.Y.Z`` from ``master``.  Use the next message::

    git tag -a vX.Y.Z -m "Tagging version X.Y.Z"

* Push the tag to the github repo::

    git push
    git push --tags

After the tag would be up, update the release notes in: https://github.com/Blosc/python-caterva/releases

* Check that the wheels are upload correctly to Pypi.

Announcing
----------

* Send an announcement to the Blosc list.  Use the ``ANNOUNCE.rst`` file as skeleton
  (or possibly as the definitive version).

* Announce in Twitter via @Blosc2 account and rejoice.


Post-release actions
--------------------

* Create new headers for adding new features in ``RELEASE_NOTES.rst``
  add this place-holder:

  XXX version-specific blurb XXX

* Edit ``VERSION`` in master to increment the version to the next
  minor one (i.e. X.Y.Z --> X.Y.(Z+1).dev0).

* Commit your changes with::

    git commit -a -m "Post X.Y.Z release actions done"
    git push


That's all folks!
