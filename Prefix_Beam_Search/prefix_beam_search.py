import math

import numpy as np


NINF = -float("inf")
EMPTY = ()
BLANK = 0


class PrefixBeamSearch(object):

    def __init__(self, logpred):
        """
        logpred:: TxC. logpred[:, BLANK] for the blank class.
        """
        self.logpred = logpred
        self.cache = init_cache(len(self.logpred))

    def search(self, width):
        t = -1
        path_new = EMPTY
        candidates_prev = [(_logsumexp2(*logpb_logpn(t, path_new, self.logpred, self.cache)), path_new)]
        class_range = range(len(self.logpred[0]))
        for t in range(len(self.logpred)):
            candidates_new = set()
            for _, path_prev in candidates_prev:
                for c in class_range:
                    path_new = path_prev if c == BLANK else path_prev + (c,)
                    if path_new in candidates_new:
                        continue
                    candidates_new.add((_logsumexp2(*logpb_logpn(t, path_new, self.logpred, self.cache)), path_new))
            candidates_prev = sorted(candidates_new)[-width:]
        for ll, path in candidates_prev:
            yield path, ll


def init_cache(T):
    cache = {t: dict() for t in range(T)}
    cache[-1] = {EMPTY: (0, NINF)}
    return cache


def logpb_logpn(t, path, logpred, cache):
    cache_t = cache[t]
    if path in cache_t:
        return cache_t[path]
    if path is EMPTY:
        logPb_t1_empty, logPn_t1_empty = cache[t - 1][EMPTY]
        ret = (logpred[t][BLANK] + logPb_t1_empty, NINF)
        cache_t[path] = ret
        return ret
    if t < 0:
        ret = (NINF, NINF)
        cache_t[path] = ret
        return ret

    logPb_t1_path, logPn_t1_path = logpb_logpn(t - 1, path, logpred, cache)
    logpred_t = logpred[t]  # optimized
    logPb_t_path = logpred_t[BLANK] + _logsumexp2(logPb_t1_path, logPn_t1_path)
    c_last = path[-1]
    logPb_t1_path1, logPn_t1_path1 = logpb_logpn(t - 1, path[:-1], logpred, cache)
    if (len(path) > 1) and path[-2] == c_last:
        logPn_t_path = logpred_t[c_last] + _logsumexp2(logPn_t1_path,                 logPb_t1_path1)
    else:
        logPn_t_path = logpred_t[c_last] + _logsumexp3(logPn_t1_path, logPb_t1_path1, logPn_t1_path1)
    ret = (logPb_t_path, logPn_t_path)
    cache_t[path] = ret
    return ret


def logsoftmax(x):
    xmax = x.max(axis=1, keepdims=True)
    x = x - xmax
    z = np.exp(x).sum(axis=1, keepdims=True)
    return x - np.log(z)


def _logsumexp2(x, y):
    m = max(x, y)
    if m <= NINF:
        return NINF
    return math.log(math.exp(x - m) + math.exp(y - m)) + m


def _logsumexp3(x, y, z):
    m = max(x, y, z)
    if m <= NINF:
        return NINF
    return math.log(math.exp(x - m) + math.exp(y - m) + math.exp(z - m)) + m


def main():
    np.random.seed(42)
    logpred = np.random.randn(60, 10000)
    logpred[:, 0] += 1/2
    logpred[:, 1] += 4
    pbs = PrefixBeamSearch(logsoftmax(logpred))
    ret = list(pbs.search(5))
    for r in ret:
        print(len(r[0]), r)


if __name__ == "__main__":
    main()
    # print(list(pbs.search(2)))
    # pbs = PrefixBeamSearch(logsoftmax(np.random.randn(100, 10000)))
    # print(list(pbs.search(10)))
