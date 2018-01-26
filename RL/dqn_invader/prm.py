#!/usr/bin/python

import collections
import logging
import random
import sys

import numpy as np


__version__ = "0.1.0"
logger = logging.getLogger(__name__)


_HeapEntry = collections.namedtuple("_HeapEntry", ["i", "v"])


class PrioritizedReplayMemory(object):

    def __init__(self, capacity, n_batch, alpha, random_state):
        assert alpha >= 0
        assert n_batch > 0
        self.capacity = capacity
        self.alpha = alpha
        self.n_batch = n_batch
        self.random_state = random_state
        self.rng = random.Random(self.random_state)
        self.heap = [None]*self.capacity
        self.index_ring_buffer = [None]*self.capacity
        self.pointer = 0
        self._partitions = _partitions_of(self.capacity, self.n_batch, self.alpha)

    def __len__(self):
        if self.is_full():
            return self.capacity
        else:
            return self.pointer

    def is_full(self):
        return self.index_ring_buffer[self.pointer] is not None

    def sort(self):
        """Sort in descending order
        """
        for i in range(len(self) - 1, 0, -1):
            self._swap(0, i)
            self._push_down(0, limit=i)
        if self.is_full():
            self.heap.reverse()
            self.index_ring_buffer.reverse()
        else:
            self.heap[:len(self)] = list(reversed(self.heap[:len(self)]))
            self.index_ring_buffer[:len(self)] = list(reversed(self.index_ring_buffer[:len(self)]))

    def push(self, x):
        assert x is not None
        he = _HeapEntry(i=self.pointer, v=x)
        if self.is_full():
            ih = self.index_ring_buffer[self.pointer]
            self.heap[ih] = he
            self._push_down(ih)
            self._push_up(ih)
        else:
            self.index_ring_buffer[self.pointer] = self.pointer
            self.heap[self.pointer] = he
            self._push_up(self.pointer)
        self.pointer = (self.pointer + 1)%self.capacity
        return self

    def to_list_heap(self, limit=None):
        if limit is None:
            limit = len(self)
        return [x.v for x in self.heap[:limit]]

    def sample(self):
        partitions = self._partitions
        if not self.is_full():
            c = len(self)/self.capacity
            partitions = [round(c*p) for p in partitions]
            # todo: interpolation with range(0, n_batch + 1)
        n_max = len(self) - 1
        ihs = [min(self.rng.randint(partitions[i], partitions[i + 1]), n_max) for i in range(len(partitions) - 1)]
        return [self.heap[ih].v for ih in ihs], ihs

    def fixup(self, i):
        self._push_down(i)
        self._push_up(i)

    def _push_up(self, i):
        while True:
            if i <= 0:
                break
            ip = _parent(i)
            if self.heap[ip].v >= self.heap[i].v:
                break
            self._swap(ip, i)
            i = ip

    def _push_down(self, i, limit=None):
        if limit is None:
            limit = len(self)
        th = _leaf_threshold(limit)
        while True:
            if i >= th:
                break
            il, ir = _left(i), _right(i)
            if (il < limit) and (self.heap[il].v > self.heap[i].v):
                imax = il
            else:
                imax = i
            if (ir < limit) and (self.heap[ir].v > self.heap[imax].v):
                imax = ir
            if imax == i:
                break
            self._swap(i, imax)
            i = imax

    def _swap(self, i, j):
        ii, ij = self.heap[i].i, self.heap[j].i
        self.heap[j], self.heap[i] = self.heap[i], self.heap[j]
        self.index_ring_buffer[ij], self.index_ring_buffer[ii] = self.index_ring_buffer[ii], self.index_ring_buffer[ij]


# todo: This is not good at all.
def _partitions_of(capacity, n_batch, alpha):
    assert 0 < n_batch <= capacity
    assert n_batch > 0
    assert alpha >= 0
    ps = 1/np.arange(1, capacity + 1)**alpha
    cs = np.cumsum(ps)
    total = cs[-1]
    ps /= total
    cs /= total
    cs[-1] = 1
    ths = list(np.linspace(0, 1, n_batch + 1))
    th_inds = [round(i*capacity/n_batch) for i in range(n_batch + 1)]
    for i_th in range(1, n_batch):
        th = ths[i_th]
        for i_c in range(th_inds[i_th - 1] + 1, th_inds[i_th]):
            if cs[i_c] > th:
                th_inds[i_th] = i_c
                break
    return th_inds


def _is_heap(x):
    if not x:
        return True
    def _impl(x, i):
        il, ir = _left(i), _right(i)
        if il < len(x):
            assert x[i] >= x[il], (x[i], x[il])
            _impl(x, il)
        if ir < len(x):
            assert x[i] >= x[ir], (x[i], x[ir])
            _impl(x, ir)
    _impl(x, 0)


def _is_sorted(xs):
    if not xs:
        return True
    x_prev = xs[0]
    for i in range(1, len(xs)):
        assert x_prev >= xs[i], (x_prev, xs[i])
        x_prev = xs[i]


def _leaf_threshold(n):
    """The minimum index of leaf nodes.

    >>> _leaf_threshold(2)
    1
    >>> _leaf_threshold(3)
    1
    >>> _leaf_threshold(4)
    2
    >>> _leaf_threshold(5)
    2
    >>> _leaf_threshold(6)
    3
    >>> _leaf_threshold(7)
    3
    """
    return n//2


def _left(i):
    """
    >>> _left(0)
    1
    """
    return 2*i + 1


def _right(i):
    """
    >>> _right(0)
    2
    """
    return 2*i + 2


def _parent(i):
    """
    >>> _parent(1)
    0
    >>> _parent(2)
    0
    >>> _parent(3)
    1
    """
    return (i - 1)//2


def _test():
    import doctest
    doctest.testmod(raise_on_error=True, verbose=True)
    n = 2000
    a = list(range(n))
    import random
    random.shuffle(a)
    prm = PrioritizedReplayMemory(n, 10, 1, 42)
    for x in a:
        prm.push(x)
    _is_heap(prm.to_list_heap())
    prm.sort()
    _is_sorted(prm.to_list_heap())
    print(prm._partitions)
    print(prm.sample())

    # push more than capacity
    b = list(range(2*n))
    random.shuffle(b)
    for x in b:
        prm.push(x)
    _is_heap(prm.to_list_heap())
    prm.sort()
    _is_sorted(prm.to_list_heap())

    # queue-ness
    prm = PrioritizedReplayMemory(3, 2, 1, 42)
    for i in range(10):
        prm.push(i)
    assert set(prm.to_list_heap()) == set([7, 8, 9]), prm.to_list_heap()

    print(_partitions_of(100, 5, 1))
    # 10000 0.482
    # 20000 0.780
    # 40000 1.788
    # 80000 3.441
    # 160000 7.609


if __name__ == "__main__":
    _test()
