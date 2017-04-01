#!/usr/bin/python

import sys

import yaml


def main(argv):
    if len(argv) != 1:
        _usage_and_exit()

    records = [None]  # padding with None
    records.extend(sorted([_parse(l) for l in sys.stdin], key=lambda r: r["j"]))
    tree = [dict(children=[]) for _ in records]
    assert records[1]["i"] == 0
    tree[0]["id"] = records[1]["i"]
    tree[0]["t"] = records[1]["ti"]
    tree[0]["m"] = records[1]["mi"]
    tree[0]["lat"] = records[1]["lati"]
    tree[0]["lon"] = records[1]["loni"]
    for j, (rec, node) in enumerate(zip(records, tree)):
        if rec is None:
            continue
        node["id"] = rec["j"]
        node["t"] = rec["tj"]
        node["m"] = rec["mj"]
        node["lat"] = rec["latj"]
        node["lon"] = rec["lonj"]
        tree[rec["i"]]["children"].append(
            dict(
                j=j,
                log_etaij=rec["log_etaij"],
                log_tij=rec["log_tij"],
                log_rij=rec["log_rij"],
                log_mi=rec["log_mi"],
            ),
        )
    yaml.dump(
        tree,
        sys.stdout,
        explicit_start=True,
        default_flow_style=False,
        width=2**31,
        indent=4,
    )


def _usage_and_exit(s=1):
    if s == 0:
        fh = sys.stdout
    else:
        fh = sys.stderr
    print('python3 {} < <zaliapin_ben_zion_2013.exe.output> > <spanning_tree.yaml>'.format(__file__), file=fh)
    exit(s)


def _parse(line):
    j, i, log_etaij, log_tij, log_rij, log_mi, tj, mj, latj, lonj, ti, mi, lati, loni = line.split()
    return dict(
        j=int(j),
        i=int(i),

        log_etaij=float(log_etaij),
        log_tij=float(log_tij),
        log_rij=float(log_rij),
        log_mi=float(log_mi),

        tj=float(tj),
        mj=float(mj),
        latj=float(latj),
        lonj=float(lonj),

        ti=float(ti),
        mi=float(mi),
        lati=float(lati),
        loni=float(loni),
    )


if __name__ == '__main__':
    main(sys.argv)
