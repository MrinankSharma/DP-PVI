import torch


class TensorBuffer(object):
    """ Lightweight class to hold series fo tensors of a given shape
    in a size adaptive tensor. Utilised via the append method.
    """

    def __init__(self, capacity, shape, dtype=torch.float32):
        """ Initialise the TensorBuffer

        :param capacity: The initial capacity of the buffer. Will double
        in size when capacity is reached
        :param shape: The shape of the Tensors to accumulate
        :param dtype: The datatype of the buffer
        """

        self._entry_shape = torch.Size(shape)
        shape = list(shape)
        self._rank = len(shape)
        self._dtype = dtype
        if not self._rank:
            raise ValueError("Shape cannot be scalar")
        shape = [capacity] + shape

        self._buffer = torch.zeros(shape, dtype=self._dtype)
        self._current_size = torch.tensor([0], dtype=torch.long)
        self._capacity = torch.tensor([capacity], dtype=torch.long)

    def append(self, value):
        """ Appends a new tensor to the end of the buffer.

        :param value: The new tensor to append
        """

        def _double_capacity():
            """ Doubles the capacity of the buffer """
            padding = torch.zeros_like(self._buffer)
            self._buffer = torch.cat([self._buffer, padding], dim=0)
            self._capacity = self._capacity * 2


        if self._current_size == self._capacity:
            _double_capacity()

        assert self._current_size < self._capacity
        assert value.shape == self._entry_shape

        self._buffer[self._current_size, :] = value
        self._current_size = self._current_size + 1

    @property
    def values(self):
        return self._buffer[:self._current_size, :]

    @property
    def current_size(self):
        return self._current_size

    @property
    def capacity(self):
        return self._capacity
