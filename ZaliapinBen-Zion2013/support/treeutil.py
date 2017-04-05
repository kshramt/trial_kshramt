def root_of(tree):
    if isinstance(tree, dict):
        return tree[min(tree.keys())]
    else:
        return tree[0]


def depth_of(tree):
    return _depth_of(root_of(tree), tree, 1)


def _depth_of(node, tree, depth):
    if not node["children"]:
        return depth
    return max(_depth_of(tree[c["j"]], tree, depth + 1) for c in node["children"])
