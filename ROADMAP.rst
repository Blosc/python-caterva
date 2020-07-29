Roadmap
=======

This document lists the main goals for the upcoming cat4py releases.


Features
--------

* *Append chunks in any order*. This will make it easier for the user to
  create arrays, since they will not be forced to use a row-wise order.

* *Update array elements*. With this, users will be able to update their
  arrays without having to make a copy.

* *Resize array dimensions*. This feature will allow Caterva to increase or
  decrease in size any dimension of the arrays.


Installation
------------

* *Build wheels*. Having the wheels will make the installation easier for the
  user.


Interoperability
----------------

* *Third-party integration*. Caterva need better integration with libraries like:

    * xarray (labeled arrays)
    * dask (computation)
    * napari (visualization)
