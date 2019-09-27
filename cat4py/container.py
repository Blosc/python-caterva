# This module is only useful to been able to change the Container class to different subclasses.
# This can't be done on a Cython extension (ext.Container).

from functools import reduce
import operator
import math
from . import container_ext as ext


class Container(ext.Container):
    pass


def prod(iterable):
   return reduce(operator.mul, iterable, 1)


def get_pshape_guess(shape, itemsize=4, suggested_size=2**20):
    """Get a guess for a reasonable pshape that is compliant with shape.

    Parameters
    ----------
    shape: tuple or list
        The shape for the underlying array.
    itemsize: int
        The itemsize of the underlying array.
    suggested_size: int
        A suggestion for the partition size.

    Return
    ------
    tuple
        The guessed pshape.
    """
    goal = math.trunc(suggested_size / itemsize)
    pshape = [1] * len(shape)
    shape = shape[::-1]
    for i, shape_i in enumerate(shape):
        current_goal = prod(pshape)
        if current_goal * shape[i] < goal:
            pshape[i] = shape[i]
            continue
        ratio = math.trunc(goal / current_goal)
        if ratio > 0:
            print(ratio, goal)
            pshape[i] = ratio
        break
    return tuple(pshape[::-1])
