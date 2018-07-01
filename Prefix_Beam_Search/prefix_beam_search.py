import math
import sys

import numpy as np
import numba


NINF = -float("inf")
NIL = ()
BLANK = 0


class PrefixBeamSearch(object):

    def __init__(self, logpred):
        """
        logpred:: TxC. logpred[:, BLANK] for the blank class.
        """
        self.logpred = logpred
        # optimized
        self.cache = {t: dict() for t in range(len(self.logpred))}
        self.cache[-1] = {NIL: (0, NINF)}

    def search(self, width):
        t = -1
        path_new = NIL
        candidates_prev = [(_logsumexp2(*self.logpb_logpn(t, path_new)), path_new)]
        for t in range(len(self.logpred)):
            candidates_new = set()
            for _, path_prev in candidates_prev:
                for c in range(len(self.logpred[0])):
                    path_new = path_prev if c == BLANK else path_prev + (c,)
                    if path_new in candidates_new:
                        continue
                    candidates_new.add((_logsumexp2(*self.logpb_logpn(t, path_new)), path_new))
            candidates_prev = sorted(candidates_new)[-width:]
        for ll, path in candidates_prev:
            yield path, ll

    def logpb_logpn(self, t, path):
        cache = self.cache[t]
        if path in cache:
            return cache[path]
        if (t < 0) and (path is not NIL):
            ret = (NINF, NINF)
            cache[path] = ret
            return ret
        logPb_t1_path, logPn_t1_path = self.logpb_logpn(t - 1, path)

        logpred_t = self.logpred[t]  # optimized
        logPb_t_path = logpred_t[BLANK] + _logsumexp2(logPb_t1_path, logPn_t1_path)
        logPn_t_path = NINF if path is NIL else logpred_t[path[-1]] + _logsumexp3(logPn_t1_path, *self.logpb_logpn(t - 1, path[:-1]))
        ret = (logPb_t_path, logPn_t_path)
        cache[path] = ret
        return ret


@numba.jit
def logsoftmax(x):
    xmax = x.max(axis=1, keepdims=True)
    x = x - xmax
    z = np.exp(x).sum(axis=1, keepdims=True)
    return x - np.log(z)


@numba.jit
def _logsumexp2(x, y):
    m = max(x, y)
    if m <= NINF:
        return NINF
    return math.log(math.exp(x - m) + math.exp(y - m)) + m


@numba.jit
def _logsumexp3(x, y, z):
    m = max(x, y, z)
    if m <= NINF:
        return NINF
    return math.log(math.exp(x - m) + math.exp(y - m) + math.exp(z - m)) + m


def _rev_list_of(cell):
    ret = []
    while True:
        if cell is NIL:
            break
        l, r = cell.l, cell.r
        ret.append(l)
        cell = r
    ret.reverse()
    return ret


if __name__ == "__main__":
    pbs = PrefixBeamSearch(logsoftmax(np.random.randn(20, 5000)))
    print(list(pbs.search(3)))
    # pbs = PrefixBeamSearch(logsoftmax(np.random.randn(100, 10000)))
    # print(list(pbs.search(10)))
