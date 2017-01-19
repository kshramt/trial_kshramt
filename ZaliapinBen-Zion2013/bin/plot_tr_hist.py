#!/usr/bin/python


import math
import sys

import matplotlib.pyplot as plt


def main(argv):
    if len(argv) != 2:
        _usage_and_exit()
    q = float(argv[1])
    f, a = plt.subplots()
    xs = []
    ys = []
    for logt, logr, logm in load(sys.stdin):
        x = logt - q*logm
        y = logr - (1 - q)*logm
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    _, _, _, handle = a.hist2d(
        xs,
        ys,
        cmap='viridis',
        bins=max(int(len(xs)**(1/3) + 1), 40),
    )
    a.set_xlabel(r'Logarithm of rescaled time, T')
    a.set_ylabel(r'Logarithm of rescaled distance, R')
    f.colorbar(
        handle,
        aspect=30,
        label="Frequency",
    )
    f.savefig(
        sys.stdout.buffer,
        format='pdf',
        transparent=True,
        bbox_inches='tight',
    )


def _usage_and_exit(s=1):
    if s == 0:
        fh = sys.stdout
    else:
        fh = sys.stderr
    print('{} <q> < <catalog.distance> > <tr_hist.pdf>'.format(__file__), file=fh)
    exit(s)


def load(fh):
    for l in fh:
        # 10000 9997 9.94753437809827012e+00 7.75936667983802675e+00 5.78020044333095484e+00 3.59203274507071191e+00
        _, _, _, logt, logr, logm, *_ = l.split()
        yield float(logt), float(logr), float(logm)


if __name__ == '__main__':
    main(sys.argv)
