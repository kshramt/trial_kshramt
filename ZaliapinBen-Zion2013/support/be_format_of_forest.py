#!/usr/bin/python

# BE format:
# Index   Time   Lat.   Long.  Magnitude   log(ETA)   log(T)   log(R)   Index_to_direct_Parent    Index_to_root_in_the_family

# If the event is the root of a tree (i.e., the first event in the family - the mainshock or the first-foreshock) the
# "Index_to_direct_Parent" would be, let's say, "-1" and Index_to_root_in_the_family = index of the event himself

# If the event would be a SINGLE:
# Index_to_direct_Parent  = -1
# Index_to_root_in_the_family = -1
# log(ETA) = 12345
# log(T) = 12345
# log(R) = 12345

# As a requirement, one event should have 0 or 1 parent (i.e., cannot have two or more different parents).

import sys

import yaml


def main(argv):
    if len(argv) != 1:
        _usage_and_exit()

    forest = yaml.load(sys.stdin)
    for tree in forest:
        root = root_of(tree)
        assert len(tree) > 0
        if len(tree) == 1:
            print(root["id"], root["t"], root["lat"], root["lon"], root["m"], 12345, 12345, 12345, -1, -1, sep="\t")
        else:
            print(root["id"], root["t"], root["lat"], root["lon"], root["m"], 12345, 12345, 12345, -1, root["id"], sep="\t")
            _print_children(root, tree, root["id"])


def _usage_and_exit(s=1):
    if s == 0:
        fp = sys.stdout
    else:
        fp = sys.stderr
    print('{} < forest.haml > forest.be_format'.format(__file__), file=fp)
    exit(s)


def _print_children(parent, tree, root_id):
    for edge in parent["children"]:
        child = tree[edge["j"]]
        print(child["id"], child["t"], child["lat"], child["lon"], child["m"], edge["log_etaij"], edge["log_tij"], edge["log_rij"], parent["id"], root_id, sep="\t")
        _print_children(child, tree, root_id)


def root_of(tree):
    if isinstance(tree, dict):
        return tree[min(tree.keys())]
    else:
        return tree[0]


if __name__ == '__main__':
    main(sys.argv)
