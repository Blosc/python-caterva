from .container_ext import _Container
import numpy as np

class Container(_Container):
    def to_numpy(self, dtype):
        return np.frombuffer(self.to_buffer(), dtype=dtype).reshape(self.shape)

    def from_numpy(self, array):
        self.from_buffer(array.shape, bytes(array))
