#!/usr/bin/python

import argparse
import sys

import yaml


def main(argv):
    parser = argparse.ArgumentParser(description='Cut long edges of a spanning three')
    parser.add_argument(
        '--cutoff-log-eta',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--tree-yaml',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--forest-yaml',
        type=str,
        required=True,
    )
    args = parser.parse_args(argv[1:])

    with open(args.tree_yaml) as fp:
        tree = yaml.load(fp)
    for node in tree:
        node["children"] = [child for child in node["children"] if child["log_etaij"] < args.cutoff_log_eta]
    seen = [False for _ in tree]
    forest = []
    for i in range(len(tree)):
        _split(i, tree, seen, forest)

    with open(args.forest_yaml, "w") as fp:
        yaml.dump(
            forest,
            sys.stdout,
            explicit_start=True,
            default_flow_style=False,
            width=2**31,
            indent=4,
        )
        print(file=fp)


def _split(i, tree, seen, forest):
    if seen[i]:
        return
    subtree = dict()
    _subtree(i, tree, seen, subtree)
    forest.append(subtree)


def _subtree(i, tree, seen, subtree):
    assert not seen[i]
    assert i not in subtree
    seen[i] = True
    subtree[i] = tree[i]
    for child in subtree[i]["children"]:
        _subtree(child["j"], tree, seen, subtree)


if __name__ == '__main__':
    main(sys.argv)
