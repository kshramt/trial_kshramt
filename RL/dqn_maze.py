#!/usr/bin/python

import argparse
import collections
import copy
import datetime
import logging
import os
import random
import sys

import torch
import numpy as np


__version__ = "0.1.0"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


var = torch.autograd.Variable


with open(__file__) as fp:
    source = fp.read()


maze = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0, -1,  0,  1, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1,  0,  0,  0,  0,  0, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1],
], dtype=float)


Transition = collections.namedtuple("Transition", ("si", "ai1", "ri1", "si1", "done"))


class DQNAgent(object):

    def __init__(
            self,
            model,
            opt,
            gamma,
            alpha,
            epsilon,
            cuda,
            replay_memory,
            random_state,
            n_batch,
            prep_s,
    ):
        self.cuda = cuda
        self.model = model.cuda() if self.cuda else model
        self.target_model = copy.deepcopy(self.model)
        # loss = torch.nn.MSELoss
        loss = torch.nn.SmoothL1Loss
        # loss = torch.nn.L1Loss
        self.loss = loss().cuda() if self.cuda else loss()
        self.opt = opt
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.replay_memory = replay_memory
        self.source = source # for record
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        assert n_batch > 0
        self.n_batch = n_batch
        self.prep_s = prep_s

    def action_of(self, si):
        # return 1
        if self.rng.rand() <= self.epsilon:
            return self.rng.randint(self.model.output.out_features)
        else:
            # LongTensor([int, int64, int, ...]) is not allowed
            return int(_argmax(self.model(var(self._float_tensor([si]), volatile=True)))[0])

    def update_target_model(self):
        """Call me in the main loop.
        """
        self.target_model = copy.deepcopy(self.model)

    def train(self, si, ai1, ri1, si1, done):
        self.replay_memory.push((self.prep_s(si), ai1, ri1, self.prep_s(si1), done))
        if self.replay_memory.filled():
            return self.optimize()

    def optimize(self):
        batch = Transition(*zip(*self.replay_memory.sample(self.n_batch)))

        # Instead of adding Q(terminal state, any action) = 0 in training data,
        # we directly use that fact to reduce approximation errors.
        self.model.eval()
        v_hat_si1 = var(torch.zeros(self.n_batch), volatile=True)
        non_final_si1 = [x for x, done in zip(batch.si1, batch.done) if not done]
        if non_final_si1:
            mask = self._byte_tensor([not x for x in batch.done])
            v_hat_si1[mask] = self.target_model(var(self._float_tensor(non_final_si1), volatile=True)).max(1)[0]
        v_hat_si1.volatile = False # used as a constant later
        q_bellman = (var(self._float_tensor(batch.ri1)) + self.gamma*v_hat_si1).view(self.n_batch, -1)

        self.model.train()
        q_pred = self.model(var(self._float_tensor(batch.si))).gather(1, var(self._long_tensor(batch.ai1).view(-1, 1)))
        q_pred_const = var(q_pred.data)
        td = q_bellman - q_pred_const
        # q_target = q_pred_const + self.alpha*td
        q_target = q_bellman # Mnih et al (2015, Nature)

        loss = self.loss(q_pred, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        q_pred = self.model(var(self._float_tensor(batch.si), volatile=True)).gather(1, var(self._long_tensor(batch.ai1).view(self.n_batch, -1)))
        # return dict(td=td, loss=loss)
        return dict(td=None, loss=loss)

    def _float_tensor(self, x, **kwargs):
        return torch.cuda.FloatTensor(x, **kwargs) if self.cuda else torch.FloatTensor(x, **kwargs)

    def _long_tensor(self, x, **kwargs):
        return torch.cuda.LongTensor(x, **kwargs) if self.cuda else torch.LongTensor(x, **kwargs)

    def _byte_tensor(self, x, **kwargs):
        return torch.cuda.ByteTensor(x, **kwargs) if self.cuda else torch.ByteTensor(x, **kwargs)


class Env(object):

    def __init__(self, maze):
        self.maze = maze
        self.state = None

    def reset(self, random_state):
        rng = np.random.RandomState(random_state)
        n, m = self.maze.shape
        while True:
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            # i = 2
            # j = 5
            state = (i, j)
            if self.maze[state] == 0:
                self.state = state
                return self.state

    def step(self, action):
        """
         1
        2 0
         3
        """
        n, m = self.maze.shape
        i, j = self.state
        if (action == 0) and (j < m - 1):
            state = (i, j + 1)
        elif (action == 1) and (i > 0):
            state = (i - 1, j)
        elif (action == 2) and (j > 0):
            state = (i, j - 1)
        elif (action == 3) and (i < n - 1):
            state = (i + 1, j)
        else:
            state = (i, j)
        self.state = state
        ri1 = self.maze[self.state]
        return self.state, ri1, bool(ri1 == 1), None


class ReplayMemory(object):

    def __init__(self, capacity, random_state):
        assert capacity > 0, capacity
        self.capacity = capacity
        # si, ai1, ri1, si1, done
        self.buffer = [None]*self.capacity
        self.pointer = 0
        self.random_state = random_state # for record
        self.rng = random.Random(self.random_state)

    def push(self, x):
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1)%self.capacity
        return self

    def sample(self, n):
        return self.rng.sample(self.buffer, n)

    def filled(self):
        return self.buffer[self.pointer] is not None


class Swish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.nn.functional.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class Log1p(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sign(input)*torch.log1p(torch.abs(input))

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


def main(argv):
    args = _parse_argv(argv[1:])
    _add_handlers(logger, args.log_file, args.log_stderr_level, args.log_file_level)
    logger.info(f"args\t{args}")
    run(args, maze)


def run(args, maze):
    torch.manual_seed(args.torch_seed)

    n_input = maze.ndim
    n_output = 2**maze.ndim
    logger.info(f"n_input, args.n_middle, n_output\t{n_input, args.n_middle, n_output}")
    namer = _make_namer()
    act = Swish
    bn = lambda : torch.nn.BatchNorm1d(num_features=args.n_middle, momentum=1e-3, affine=False)
    # act = Log1p
    model = torch.nn.Sequential(collections.OrderedDict((
        (namer("fc"), torch.nn.Linear(n_input, args.n_middle)),
        (namer("ac"), act()),
        # (namer("bn"), bn()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        # (namer("bn"), bn()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        # (namer("bn"), bn()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        # (namer("bn"), bn()),
        ("output", torch.nn.Linear(args.n_middle, n_output)),
    )))
    def _init(m):
        if type(m) == torch.nn.Linear:
           torch.nn.init.kaiming_uniform(m.weight.data)
           m.bias.data.fill_(0)
    model.apply(_init)
    # opt = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=1e-1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    agent = DQNAgent(
        model=model, opt=opt, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma,
        replay_memory=ReplayMemory(capacity=args.n_replay_memory, random_state=args.replay_memory_seed),
        cuda=torch.cuda.is_available(),
        random_state=args.agent_seed,
        n_batch=args.n_batch,
        prep_s=lambda s: (s[0]/maze.shape[0], s[1]/maze.shape[1]),
    )

    env = Env(maze)
    episode_result_list = []
    for i_episode, env_seed in zip(range(args.n_episodes), _seed_generator(args.env_seed)):
        logger.info(f"i_episode\t{i_episode}")
        si = env.reset(env_seed)
        step_result_list = []
        for i_step in range(args.n_steps):
            ai1 = agent.action_of(si)
            si1, ri1, done, debug_info = env.step(ai1)
            metric = agent.train(si, ai1, ri1, si1, done)
            if i_step%args.n_log_steps == 0:
                step_result_list.append(dict(i_step=i_step, si=si, ai1=ai1, ri1=ri1, si1=si1, metric=metric))
                if metric is not None:
                    logger.info(f"loss\t{metric['loss'].data.numpy()[0]}")
            si = si1
            if done:
                episode_result_list.append(dict(i_episode=i_episode, env_seed=env_seed, step_result_list=step_result_list))
                break
        if i_episode%args.n_target_update_episodes == 0:
            agent.update_target_model()
        if i_episode%10 == 0 and agent.replay_memory.filled():
            pass
            print_q(agent, *maze.shape)


def print_q(agent, n, m, fp=sys.stderr):
    agent.model.eval()
    print(file=fp)
    for i in range(n):
        for j in range(m):
            x, y = agent.prep_s((i, j))
            print("    ", end="", file=fp)
            q = agent.model(var(agent._float_tensor([[x, y]]), volatile=True)).data.numpy()
            print(f"{q[0, 1]:6.2f} ", end="", file=fp)
            print("    ", end="", file=fp)
        print(file=fp)
        for j in range(m):
            x, y = agent.prep_s((i, j))
            q = agent.model(var(agent._float_tensor([[x, y]]), volatile=True)).data.numpy()
            print(f"{q[0, 2]:6.2f} ", end="", file=fp)
            print(" ", end="", file=fp)
            print(f"{q[0, 0]:6.2f} ", end="", file=fp)
        print(file=fp)
        for j in range(m):
            x, y = agent.prep_s((i, j))
            q = agent.model(var(agent._float_tensor([[x, y]]), volatile=True)).data.numpy()
            print("    ", end="", file=fp)
            print(f"{q[0, 3]:6.2f} ", end="", file=fp)
            print("    ", end="", file=fp)
        print(file=fp)

        print(file=fp)
    # import time
    # time.sleep(2)


def _parse_argv(argv):
    logger.debug(f"argv\t{argv}")
    doc = f"""
    {__file__}
    """
    parser = argparse.ArgumentParser(doc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lr", required=True, type=float, help="Learning rate.")
    parser.add_argument("--alpha", required=True, type=float, help="α for TD error.")
    parser.add_argument("--epsilon", required=True, type=float, help="ε-greedy.")
    parser.add_argument("--gamma", required=True, type=float, help="γ for discounted reward.")
    parser.add_argument("--log-td", required=True, type=str, help="Log file for TD errors.")
    parser.add_argument("--n-steps", required=True, type=int, help="Number of maximum steps per episode.")
    parser.add_argument("--n-log-steps", required=True, type=int, help="Record logs per this steps")
    parser.add_argument("--n-episodes", required=True, type=int, help="Number of episodes to run.")
    parser.add_argument("--n-batch", required=True, type=int, help="Batch size.")
    parser.add_argument("--n-replay-memory", required=True, type=int, help="Capacity of the replay memory.")
    parser.add_argument("--n-target-update-episodes", required=True, type=int, help="Number of episodes to update the target network.")
    parser.add_argument("--n-middle", required=True, type=int, help="Number of units in a hidden layer.")
    parser.add_argument("--replay-memory-seed", required=True, type=int, help="Random state for minibatch.")
    parser.add_argument("--env-seed", required=True, type=int, help="Seed to reset the environment.")
    parser.add_argument("--agent-seed", required=True, type=int, help="Seed for the agent.")
    parser.add_argument("--torch-seed", required=True, type=int, help="Seed for PyTorch.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-stderr-level", default="info", type=lambda x: getattr(logging, x.upper()), help="Set log level for stderr.")
    parser.add_argument("--log-file-level", default="debug", type=lambda x: getattr(logging, x.upper()), help="Set log level for the log file.")
    parser.add_argument("--log-file", default=os.path.join("log", datetime.datetime.now().strftime("%y%m%d%H%M%S") + "_" + str(os.getpid()) + "_" + os.path.basename(__file__) + ".log"), help="Set log file.")
    args = parser.parse_args(argv)
    logger.debug(f"args\t{args}")
    assert args.n_batch <= args.n_replay_memory, (args.n_batch, args.n_replay_memory)
    return args


def _argmax(pred):
    return pred.max(1)[1].data.numpy()


def _seed_generator(random_state):
    rng = np.random.RandomState(random_state)
    while True:
        yield rng.randint(2**32)


def _make_namer():
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


def _add_handlers(logger, path, level_stderr, level_path):
    fmt = logging.Formatter("%(levelname)s\t%(process)d\t%(asctime)s\t%(filename)s\t%(funcName)s\t%(lineno)d\t%(message)s")

    hdl = logging.StreamHandler(sys.stderr)
    # hdl = logging.StreamHandler(sys.stdout)
    hdl.setFormatter(fmt)
    logger.addHandler(hdl)
    hdl.setLevel(level_stderr)
    logger.addHandler(hdl)

    _mkdir(_dirname(path))
    hdl = logging.FileHandler(path)
    hdl.setFormatter(fmt)
    hdl.setLevel(level_path)
    logger.addHandler(hdl)

    logger.info(f"log file\t{path}")
    return logger


def _mkdir(path):
    os.makedirs(path, exist_ok=True)


def _dirname(path):
    return os.path.dirname(path) or os.path.curdir


if __name__ == "__main__":
    main(sys.argv)
