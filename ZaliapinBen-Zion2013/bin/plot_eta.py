#!/usr/bin/python


import sys
import math

import matplotlib.pyplot as plt


def main(argv):
    if len(argv) != 1:
        usage_and_exit()
    f, a = plt.subplots()
    log_etas = [x for x in load(sys.stdin) if math.isfinite(x)]
    a.hist(
        log_etas,
        color='lightgray',
        bins=int(math.sqrt(len(log_etas)) + 1),
        histtype='stepfilled',
        linewidth=0.5,
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
    print('{} < <catalog.distance> > <eta.pdf>'.format(__file__), file=fh)
    exit(s)


def load(fh):
    for l in fh:
        # 10000 9997 9.94753437809827012e+00 7.75936667983802675e+00 5.78020044333095484e+00 3.59203274507071191e+00
        _, _, log_eta, *_ = l.split()
        yield float(log_eta)


if __name__ == '__main__':
    main(sys.argv)
