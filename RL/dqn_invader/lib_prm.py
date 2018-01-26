#!/usr/bin/python

import collections
import copy
import logging
import math
import os
import random
import sys

import torch
from torch.autograd import Variable as var
import gym
import numpy as np

import prm


inf = float("inf")


__version__ = "0.1.0"
logger = logging.getLogger(__name__)


# First, choose `action` based on `state`.
# Then, receive `reward_next`, `state_next`, and `done`.
Transition = collections.namedtuple("Transition", ["state", "action", "reward_next", "state_next", "done"])


class ReplayMemoryEntry(object):

    def __init__(self, t, p):
        self.t = t
        self.p = p

    def __lt__(self, other):
        return self.p < other.p

    def __ge__(self, other):
        return self.p >= other.p

    def __repr__(self):
        return f"ReplayMemoryEntry(t={self.t}, p={self.p})"


class SpaceInvaderWrapper(object):

    def __init__(self, n_for_max, n_concatenate):
        assert n_for_max > 0
        assert n_concatenate > 0
        self.n_for_max = n_for_max
        self.n_concatenate = n_concatenate
        self.buffer = RingBuffer(self.n_concatenate)
        self.env = gym.make("SpaceInvaders-v0").unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        state = self.env.reset()
        for _ in range(len(self.buffer)):
            self.buffer.push(state.copy())
        return self.to_state()

    def to_state(self):
        return np.array(self.buffer.to_list())

    @staticmethod
    def prep(self, img):
        img.mean(axis=2)

    @staticmethod
    def gray(img):
        return img.mean(axis=2)


class Mean0(torch.nn.Module):

    def forward(self, input):
        return input - input.mean(1).view(-1, 1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Flatten(torch.nn.Module):

    def __init__(self, n_input):
        assert n_input > 0
        super().__init__()
        self.n_input = n_input

    def forward(self, input):
        return input.view(-1, self.n_input)

    def __repr__(self):
        return self.__class__.__name__ + f"(n_input={self.n_input})"


class Agent(object):
    """
    Double DQN.
    Supports both TD and Mnih+2015 update.
    """

    def __init__(
            self,
            model,
            replay_memory,
            n_batch,
            cuda,
            alpha,
            gamma,
            loss,
            opt,
            dqn_mode,
            td_mode,
    ):
        assert 0 < n_batch <= replay_memory.capacity
        assert dqn_mode in ("dqn", "doubledqn")
        assert td_mode in ("td", "mnih2015")
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.replay_memory = replay_memory
        self.n_batch = n_batch
        self.cuda = cuda
        self.alpha = alpha
        self.gamma = gamma
        self.loss = loss
        self.opt = opt
        self.dqn_mode = dqn_mode
        self.td_mode = td_mode
        self.ftn = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.ltn = torch.cuda.LongTensor if cuda else torch.LongTensor
        self.btn = torch.cuda.ByteTensor if cuda else torch.ByteTensor

    def act(self, state):
        self.model.eval()
        q = self.model(var(self.ftn(state).view(1, -1), volatile=True))
        return int(q.max(1)[1].data.numpy()[0])

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def push(self, state, action, reward_next, state_next, done):
        self.replay_memory.push(ReplayMemoryEntry(t=Transition(state=state, action=action, reward_next=reward_next, state_next=state_next, done=done), p=inf))

    def train(self):
        samples, ihs = self.replay_memory.sample()
        transitions = [s.t for s in samples]
        batch = Transition(*zip(*transitions))
        reward_next = var(self.ftn(batch.reward_next).view(-1, 1))
        vhat = self.vhat(batch)
        q_bellman = reward_next + self.gamma*vhat
        self.model.train()
        q_pred = self.model(var(self.ftn(batch.state))).gather(1, var(self.ltn(batch.action).view(-1, 1)))
        q_pred_const = var(q_pred.data)
        td = q_bellman - q_pred_const
        if self.td_mode == "mnih2015":
            q_target = q_bellman
        elif self.td_mode == "td":
            q_target = q_bellman + self.alpha*td
        else:
            raise NotImplementedError(f"Unsupported self.q_target_mode: {self.q_target_mode}")
        loss = self.loss(q_pred, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.model.eval()
        for sample, ih, _td in zip(samples, ihs, td.data.numpy().reshape(-1)):
            sample.p = abs(_td)
            self.replay_memory.fixup(ih)
        return dict(loss=loss, td=td)

    def vhat(self, batch):
        ret = var(torch.zeros((self.n_batch, 1)), volatile=True)
        non_final_state_next = [state_next for state_next, done in zip(batch.state_next, batch.done) if not done]
        if non_final_state_next:
            mask = self.btn([not done for done in batch.done])
            non_final_state_next = var(self.ftn(non_final_state_next), volatile=True)
            self.target_model.eval()
            q = self.target_model(non_final_state_next)
            if self.dqn_mode == "dqn":
                ret[mask] = q.max(1)[0].view(-1, 1)
            elif self.dqn_mode == "doubledqn":
                self.model.eval()
                action_next = self.model(non_final_state_next).max(1)[1].view(-1, 1)
                ret[mask] = q.gather(1, action_next)
            else:
                raise NotImplementedError(f"Unsupported self.dqn_mode: {self.dqn_mode}")
        ret.volatile = False
        return ret

    def ftn(self, x):
        if self.cuda:
            return torch.cuda.FloatTensor(x)
        else:
            return torch.FloatTensor(x)

    def ltn(self, x):
        if self.cuda:
            return torch.cuda.LongTensor(x)
        else:
            return torch.LongTensor(x)

    def btn(self, x):
        if self.cuda:
            return torch.cuda.ByteTensor(x)
        else:
            return torch.ByteTensor(x)


class RingBuffer(object):

    def __init__(self, capacity):
        assert capacity > 0
        self.capacity = capacity
        self.buffer = [None]*self.capacity
        self.pointer = 0

    def __len__(self):
        if self.full():
            return self.capacity
        else:
            return self.pointer

    def full(self):
        return self.buffer[self.pointer] is not None

    def push(self, x):
        assert x is not None
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1)%self.capacity

    def to_list(self):
        if self.full():
            return [self.buffer[(self.pointer + i)%self.capacity] for i in range(self.capacity)]
        else:
            return self.buffer[:self.pointer]


class ReplayMemory(RingBuffer):
    """
    >>> rm = ReplayMemory(10000, 42)
    >>> for i in range(100): rm.push(i)
    >>> rm.sample(3)
    [81, 14, 3]
    """

    def __init__(self, capacity, random_state):
        super().__init__(capacity)
        self.rng = random.Random(random_state)

    def sample(self, n):
        assert 0 < n <= len(self)
        if self.full():
            return self.rng.sample(self.buffer, n)
        else:
            return self.rng.sample(self.buffer[:len(self)], n)


class Model(torch.nn.Module):

    def __init__(self, feature, value, advantage):
        super().__init__()
        self.feature = feature
        self.value = value
        self.advantage = advantage

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a


def make_cnn_feature(in_channels):
    namer = make_namer()
    return torch.nn.Sequential(collections.OrderedDict([
        # 210 160

        (namer("conv"), torch.nn.Conv2d(in_channels, 10, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        # 104 79

        (namer("conv"), torch.nn.Conv2d(10, 15, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(15, affine=False)),
        # 51 39

        (namer("conv"), torch.nn.Conv2d(15, 20, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(20, affine=False)),
        # 51 39

        (namer("conv"), torch.nn.Conv2d(20, 25, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(25, affine=False)),
        # 25 19

        (namer("conv"), torch.nn.Conv2d(25, 20, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(20, affine=False)),
        # 12 9

        (namer("conv"), torch.nn.Conv2d(20, 15, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(15, affine=False)),
        # 5 4

        (namer("conv"), torch.nn.Conv2d(15, 10, 3, stride=2)),
        (namer("act"), torch.nn.ReLU()),
        (namer("bn"), torch.nn.BatchNorm2d(10, affine=False)),
        # 2 1

        (namer("flatten"), Flatten(20)),
    ]))


def init_model(m):
    if type(m) == torch.nn.Linear:
       torch.nn.init.kaiming_uniform(m.weight.data)
       m.weight.data.div_(math.sqrt(2))
       m.bias.data.fill_(0)
    elif type(m) == torch.nn.Conv2d:
        size = m.weight.data.size()
        n_in = size[1]*size[2]*size[3]
        bound = math.sqrt(6/n_in)
        m.weight.data.uniform_(-bound, bound)
        m.bias.data.fill_(0)


def make_namer():
    tbl = dict()
    seen = set()
    def namer(name):
        if name in tbl:
            tbl[name] += 1
        else:
            tbl[name] = 0
        ret = f"{name}_{tbl[name]}"
        assert ret not in seen
        seen.add(ret)
        return ret
    return namer


def _add_handlers(logger, path, level_stderr=logging.INFO):
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)s\t%(process)d\t%(asctime)s\t%(filename)s\t%(funcName)s\t%(lineno)d\t%(message)s")

    hdl = logging.StreamHandler(sys.stderr)
    hdl.setFormatter(fmt)
    hdl.setLevel(level_stderr)
    logger.addHandler(hdl)

    _mkdir(_dirname(path))
    hdl = logging.FileHandler(path)
    hdl.setFormatter(fmt)
    hdl.setLevel(logging.DEBUG)
    logger.addHandler(hdl)

    logger.info(f"log file\t{path}")
    return logger


def _conf_of(**kwargs):
    return collections.namedtuple("_Conf", kwargs.keys())(**kwargs)


def _mkdir(path):
    os.makedirs(path, exist_ok=True)


def _dirname(path):
    return os.path.dirname(path) or os.path.curdir
