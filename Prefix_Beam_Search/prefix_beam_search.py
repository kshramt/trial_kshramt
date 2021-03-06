import math

import cython
import numpy as np


NINF: cython.double = -float("inf")
EMPTY = ()
BLANK: int = 0


class PrefixBeamSearch(object):

    def __init__(self, logpred):
        """
        logpred:: TxC. logpred[:, BLANK] for the blank class.
        """
        self.logpred = logpred
        self.cache = init_cache(len(self.logpred))

    def search(self, width: int):
        t: int = -1
        path_new = EMPTY
        candidates_prev = [(logpb_logpn_logp(t, path_new, self.logpred, self.cache)[2], path_new)]
        class_range = range(len(self.logpred[0]))
        for t in range(len(self.logpred)):
            candidates_new = set()
            for _, path_prev in candidates_prev:
                c: int
                for c in class_range:
                    path_new = path_prev if c == BLANK else path_prev + (c,)
                    if path_new in candidates_new:
                        continue
                    candidates_new.add((logpb_logpn_logp(t, path_new, self.logpred, self.cache)[2], path_new))
            candidates_prev = sorted(candidates_new)[-width:]
        return [(path, ll) for ll, path in candidates_prev]


def init_cache(T):
    cache = {t: dict() for t in range(T)}
    cache[-1] = {EMPTY: (0, NINF, 0)}
    return cache


def logpb_logpn_logp(t: int, path, logpred, cache):
    cache_t = cache[t]
    if path in cache_t:
        return cache_t[path]
    if path is EMPTY:
        logPb_t1_empty, logPn_t1_empty, _ = cache[t - 1][EMPTY]
        logPb_t_empty = logpred[t][BLANK] + logPb_t1_empty
        ret = (logPb_t_empty, NINF, logPb_t_empty)
        cache_t[path] = ret
        return ret
    if t < 0:
        ret = (NINF, NINF, NINF)
        cache_t[path] = ret
        return ret

    logPb_t1_path, logPn_t1_path, logP_t1_path = logpb_logpn_logp(t - 1, path, logpred, cache)
    logpred_t = logpred[t]  # optimized
    logPb_t_path = logpred_t[BLANK] + logP_t1_path
    c_last = path[-1]
    logPb_t1_path1, logPn_t1_path1, logP_t1_path1 = logpb_logpn_logp(t - 1, path[:-1], logpred, cache)
    if (len(path) > 1) and path[-2] == c_last:
        logPn_t_path = logpred_t[c_last] + _logsumexp2(logPn_t1_path, logPb_t1_path1)
    else:
        logPn_t_path = logpred_t[c_last] + _logsumexp2(logPn_t1_path, logP_t1_path1)
    ret = (logPb_t_path, logPn_t_path, _logsumexp2(logPb_t_path, logPn_t_path))
    cache_t[path] = ret
    return ret


def logsoftmax(x):
    xmax = x.max(axis=1, keepdims=True)
    x = x - xmax
    z = np.exp(x).sum(axis=1, keepdims=True)
    return x - np.log(z)


def _logsumexp2(x: cython.double, y: cython.double):
    if x > y:
        return x + math.log1p(math.exp(y - x))
    elif y == NINF:
        return NINF
    else:
        return y + math.log1p(math.exp(x - y))


def main():
    np.random.seed(42)
    logpred = np.random.randn(100, 200)
    logpred[:, 0] += 1/2
    logpred[:, 1] += 4
    pbs = PrefixBeamSearch(logsoftmax(logpred))
    ret = pbs.search(5)
    for r in ret:
        print(len(r[0]), r)


if __name__ == "__main__":
    main()
