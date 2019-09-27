# This module is only useful to been able to change the Container class to different subclasses.
# This can't be done on a Cython extension (ext.Container).
from . import container_ext as ext


class Container(ext.Container):
    pass
