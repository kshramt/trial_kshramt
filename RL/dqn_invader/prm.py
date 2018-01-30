#!/usr/bin/python

import collections
import logging
import random

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
        self._reverse()

    def push(self, x):
        assert x is not None
        he = _HeapEntry(i=self.pointer, v=x)
        if self.is_full():
            ih = self.index_ring_buffer[self.pointer]
            self.heap[ih] = he
            self.fixup(ih)
        else:
            self.index_ring_buffer[self.pointer] = self.pointer
            self.heap[self.pointer] = he
            self._push_up(self.pointer)
        self.pointer = (self.pointer + 1)%self.capacity
        return self

    def heap_to_list(self, limit=None):
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

    def _reverse(self):
        n = len(self)
        if n < 2:
            return
        # 0 1 2 3 4 5 6 7
        # - - 0 0 1 2 3 3
        nm1 = n - 1
        for i in range(n//2):
            self._swap(i, nm1 - i)

    def _swap(self, i, j):
        ii, ji = self.heap[i].i, self.heap[j].i
        self.heap[j], self.heap[i] = self.heap[i], self.heap[j]
        self.index_ring_buffer[ji], self.index_ring_buffer[ii] = self.index_ring_buffer[ii], self.index_ring_buffer[ji]


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

    def let(f):
        f()

    @let
    def _():
        n = 2000
        a = list(range(n))
        import random
        random.shuffle(a)
        prm = PrioritizedReplayMemory(n, 10, 1, 42)
        for x in a:
            prm.push(x)
        _is_heap(prm.heap_to_list())
        prm.sort()
        _is_sorted(prm.heap_to_list())

    @let
    def push_more_than_capacity():
        n = 2000
        prm = PrioritizedReplayMemory(n, 10, 1, 42)
        b = list(range(10*n))
        random.shuffle(b)
        for x in b:
            prm.push(x)
        _is_heap(prm.heap_to_list())
        prm.sort()
        _is_sorted(prm.heap_to_list())

    @let
    def queue_ness():
        capacity = 40
        prm = PrioritizedReplayMemory(capacity, 10, 1, 42)
        for _ in range(10):
            a = [random.random() for _ in range(200)]
            for x in a:
                prm.push(x)
            assert set(prm.heap_to_list()) == set(a[-capacity:]), prm.heap_to_list()

    @let
    def queue_ness_with_sort():
        capacity = 40
        prm = PrioritizedReplayMemory(capacity, 10, 1, 42)
        for i in range(20):
            print(i)
            if i%3 == 0:
                print("halved")
                n_push = capacity//2
            else:
                n_push = capacity*3
            a = [random.random() for _ in range(n_push)]
            for x in a:
                prm.push(x)
            if i%2 == 0:
                print("sorted")
                prm.sort()
            if i%3 != 0:
                assert set(prm.heap_to_list()) == set(a[-capacity:]), f"{prm.heap_to_list()}\n{a[-capacity:]}"

    # 10000 0.482
    # 20000 0.780
    # 40000 1.788
    # 80000 3.441
    # 160000 7.609


if __name__ == "__main__":
    _test()
