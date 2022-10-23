#!/usr/bin/python


import sys

import numpy as np


def main(argv):
    if len(argv) != 3:
        _usage_and_exit()
    actual = np.loadtxt(argv[1])
    expected = np.loadtxt(argv[2])
    if np.allclose(actual, expected):
        exit(0)
    else:
        print(actual, file=sys.stderr)
        print(expected, file=sys.stderr)
        print(actual - expected, file=sys.stderr)
        exit(1)


def _usage_and_exit(s=1):
    if s == 0:
        fh = sys.stdout
    else:
        fh = sys.stderr
    print('{} <actual> <expected> '.format(__file__), file=fh)
    exit(s)


if __name__ == '__main__':
    main(sys.argv)
