#!/usr/bin/python3

import logging
import math
import sys


__version__ = "0.1.0"
logger = logging.getLogger(__name__)


class Node:

    def __init__(self, parent, n_actions, done, reward):
        assert n_actions > 0, n_actions
        self.parent = parent
        self.n_actions = n_actions
        self.done = done
        self.reward = reward
        self.children = [None for _ in range(self.n_actions)]
        self.mu = 0
        self.sigma2 = 0
        self.n = 1

    def pp(self, prefix="", file=sys.stdout):
        print(prefix, f"{self.n}\t{self.mu}\t{self.sigma2}", sep="", file=file)
        for child in self.children:
            if child:
                child.pp(prefix=prefix + "  ", file=file)


def thompson_search(env, obs, n_actions, n_try, rng):
    assert 0 < n_actions < n_try, (n_actions, n_try)
    root = Node(parent=None, n_actions=n_actions, done=False, reward=0)
    actions = list(range(n_actions))

    for i_try in range(n_try):
        leaf, leaf_obs = _tree_policy(env, obs, root, rng)
        reward = _default_policy(leaf_obs, leaf.done, env, actions, rng)
        _backup(leaf, reward)
    return _best_action_of(root), root


def _tree_policy(env, obs, root, rng):
    node = root
    while not node.done:
        for action, child in enumerate(node.children):
            if child is None:
                obs, reward, done, info = env(obs, action)
                node.children[action] = Node(parent=node, n_actions=node.n_actions, done=done, reward=reward)
                return node.children[action], obs
        action = _sample_action(node, rng)
        node = node.children[action]
        obs, node.reward, node.done, info = env(obs, action)
    # Avoid holding `obs`s in non-leaf nodes to reduce memory usage.
    return node, obs


def _default_policy(obs, done, env, actions, rng):
    reward_total = 0
    while not done:
        obs, reward, done, info = env(obs, rng.choice(actions))
        reward_total += reward
    return reward_total


def _backup(node, reward):
    while node:
        reward += node.reward
        dmu = (reward - node.mu)/(node.n + 1)
        dsigma2 = ((reward - node.mu)**2 - node.sigma2)/(node.n + 1) - dmu**2
        node.mu += dmu
        node.sigma2 += dsigma2
        node.n += 1
        node = node.parent


def _sample_action(node, rng):
    action, _ = _argmax(rng.gauss(child.mu, math.sqrt(child.sigma2)) for child in node.children)
    assert action > -1
    return action


def _best_action_of(node):
    action, _ = _argmax(child.mu for child in node.children)
    assert action > -1
    return action


def _argmax(xs):
    i_best = -1
    x_best = -float("inf")
    for i, x in enumerate(xs):
        if x > x_best:
            x_best = x
            i_best = i
    return i_best, x_best


def _test():
    import random

    def env(obs, action):
        return None, random.random() + int(action), random.random() < 0.5, None
    # No syntax nor arity error.
    action, root = thompson_search(env, None, 2, 20000, random.Random(42))
    print(root.children[0].n, root.children[1].n)
    # root.pp()


if __name__ == "__main__":
    _test()
