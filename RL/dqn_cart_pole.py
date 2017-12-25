#!/usr/bin/python

import argparse
import collections
import copy
import datetime
import itertools
import logging
import os
import random
import sys

import torch
import gym
import numpy as np


__version__ = "0.1.0"
logger = logging.getLogger()


var = torch.autograd.Variable


with open(__file__) as fp:
    source = fp.read()


Transition = collections.namedtuple("Transition", ("si", "ai1", "ri1", "si1", "done"))


class DQNAgent(object):

    def __init__(
            self,
            alpha,
            cuda,
            dqn_mode,
            epsilon,
            gamma,
            model,
            n_batch,
            opt,
            q_target_mode,
            random_state,
            replay_memory,
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
        self.q_target_mode = q_target_mode
        self.dqn_mode = dqn_mode

    def action_of(self, si):
        if self.rng.rand() <= self.epsilon:
            return self.rng.randint(self.model.output.out_features)
        else:
            # LongTensor([int, int64, int, ...]) is not allowed
            self.model.eval()
            return int(self.model(var(self._float_tensor([si]), volatile=True)).max(1)[1].data.numpy()[0])

    def update_target_model(self):
        """Call me in the main loop.
        """
        self.target_model = copy.deepcopy(self.model)

    def train(self, si, ai1, ri1, si1, done):
        self.replay_memory.push((si, ai1, ri1, si1, done))
        if self.replay_memory.filled():
            return self.optimize()

    def optimize(self):
        batch = self.replay_memory.sample(self.n_batch - 1)
        batch.append(self.replay_memory.buffer[self.replay_memory.pointer])
        batch = Transition(*zip(*batch))
        v_hat_si1 = self._v_hat_si1_of(batch)
        q_bellman = (var(self._float_tensor(batch.ri1)) + self.gamma*v_hat_si1).view(self.n_batch, -1)

        self.model.train()
        q_pred = self.model(var(self._float_tensor(batch.si))).gather(1, var(self._long_tensor(batch.ai1).view(-1, 1)))
        q_pred_const = var(q_pred.data)
        td = q_bellman - q_pred_const
        if self.q_target_mode == "mnih2015":
            q_target = q_bellman # Mnih et al (2015, Nature)
        elif self.q_target_mode == "td":
            q_target = q_pred_const + self.alpha*td
        else:
            raise ValueError(f"Unsupported self.q_target_mode: {self.q_target_mode}")

        loss = self.loss(q_pred, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(td=td, loss=loss)

    def _v_hat_si1_of(self, batch):
        self.model.eval()
        # Instead of adding Q(terminal state, any action) = 0 in training data,
        # we directly use that fact to reduce approximation errors.
        v_hat_si1 = var(torch.zeros(self.n_batch), volatile=True)
        non_final_si1 = [x for x, done in zip(batch.si1, batch.done) if not done]
        if non_final_si1:
            mask = self._byte_tensor([not x for x in batch.done])
            if self.dqn_mode == "dqn":
                self.target_model.eval()
                v_hat_si1[mask] = self.target_model(var(self._float_tensor(non_final_si1), volatile=True)).max(1)[0]
            elif self.dqn_mode == "doubledqn":
                self.model.eval()
                actions = self.model(var(self._float_tensor(non_final_si1), volatile=True)).max(1)[1]
                self.target_model.eval()
                v_hat_si1[mask] = self.target_model(var(self._float_tensor(non_final_si1), volatile=True)).gather(1, actions.view(-1, 1))
            else:
                raise ValueError(f"Unsupported self.dqn_mode: {self.dqn_mode}")
        v_hat_si1.volatile = False # used as a constant later
        return v_hat_si1

    def _float_tensor(self, x, **kwargs):
        return torch.cuda.FloatTensor(x, **kwargs) if self.cuda else torch.FloatTensor(x, **kwargs)

    def _long_tensor(self, x, **kwargs):
        return torch.cuda.LongTensor(x, **kwargs) if self.cuda else torch.LongTensor(x, **kwargs)

    def _byte_tensor(self, x, **kwargs):
        return torch.cuda.ByteTensor(x, **kwargs) if self.cuda else torch.ByteTensor(x, **kwargs)


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
        # if self.buffer[self.pointer] is None:
        #     self.buffer[self.pointer] = x
        #     self.pointer = (self.pointer + 1)%self.capacity
        # else:
        #     self.pointer = (self.pointer + self.rng.randint(0, self.capacity - 1))%self.capacity
        #     self.buffer[self.pointer] = x
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


class Exp(torch.nn.Module):

    def forward(self, input):
        return torch.exp(input)

    def __repr__(self):
        return self.__class__.__name__ + " ()"


class Scale(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.logscale = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, input):
        return input*torch.exp(self.logscale)

    def __repr__(self):
        return self.__calss__.__name__ + " ()"


def main(argv):
    args = _parse_argv(argv[1:])
    _add_handlers(logger, args.log_file, args.log_stderr_level, args.log_file_level)
    logger.info(f"args\t{args}")
    env = gym.make("CartPole-v1").unwrapped
    run(args, env)


def run(args, env):
    torch.manual_seed(args.torch_seed)

    n_input = env.observation_space.shape[0]
    n_output = env.action_space.n
    logger.info(f"n_input, args.n_middle, n_output\t{n_input, args.n_middle, n_output}")
    namer = _make_namer()
    act = Swish
    # bn = lambda : torch.nn.BatchNorm1d(num_features=args.n_middle, momentum=1e-4, affine=True)
    # act = Log1p
    model = torch.nn.Sequential(collections.OrderedDict((
        (namer("fc"), torch.nn.Linear(n_input, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        ("output", torch.nn.Linear(args.n_middle, n_output)),

        # (namer("fc"), torch.nn.Linear(args.n_middle, n_output)),
        # ("output", Exp()),

        # (namer("fc"), torch.nn.Linear(args.n_middle, n_output)),
        # ("output", Scale()),

        # (namer("fc"), torch.nn.Linear(args.n_middle, n_output)),
        # ("output", torch.nn.BatchNorm1d(num_features=n_output, momentum=1e-3, affine=True)),
    )))
    # model.output.out_features = n_output
    def _init(m):
        if type(m) == torch.nn.Linear:
           torch.nn.init.kaiming_uniform(m.weight.data)
           m.bias.data.fill_(0)
    model.apply(_init)
    # opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    agent = DQNAgent(
        alpha=args.alpha,
        cuda=torch.cuda.is_available(),
        dqn_mode=args.dqn_mode,
        epsilon=1,
        gamma=args.gamma,
        model=model,
        n_batch=args.n_batch,
        opt=opt,
        q_target_mode=args.q_target_mode,
        random_state=args.agent_seed,
        replay_memory=ReplayMemory(capacity=args.n_replay_memory, random_state=args.replay_memory_seed),
    )

    episode_result_list = []
    with open(args.dat_file, "w") as fp:
        for i_episode, env_seed in zip(range(args.n_episodes), _seed_generator(args.env_seed)):
            if agent.replay_memory.filled():
                agent.epsilon = max(agent.epsilon - (1 - args.epsilon)/args.n_epsilon_decay, args.epsilon)
            logger.info(f"i_episode\t{i_episode}\t{agent.epsilon}")
            env.np_random = np.random.RandomState(env_seed)
            si = env.reset()
            step_result_list = []
            for i_step in itertools.count():
                if agent.replay_memory.filled():
                    ai1 = agent.action_of(si)
                else:
                    eps, agent.epsilon = agent.epsilon, 1
                    ai1 = agent.action_of(si)
                    agent.epsilon = eps
                si1, ri1, done, debug_info = env.step(ai1)
                metric = agent.train(si, ai1, ri1, si1, done)
                if i_step%args.n_log_steps == 0 and (metric is not None):
                    metric["td"] = np.mean(metric["td"].data.numpy()**2)
                    step_result_list.append(dict(i_step=i_step, si=si, ai1=ai1, ri1=ri1, si1=si1, metric=metric))
                    logger.info(f"loss, mean(td^2)\t{metric['loss'].data.numpy()[0]}\t{metric['td']}")
                si = si1
                if done or (i_step > args.n_steps):
                    print(i_episode, i_step, sep="\t", file=fp)
                    fp.flush()
                    episode_result_list.append(dict(i_episode=i_episode, n_steps=i_step, env_seed=env_seed, step_result_list=step_result_list))
                    break
            if i_episode%args.n_target_update_episodes == 0:
                agent.update_target_model()
            if i_episode%10 == 0 and agent.replay_memory.filled():
                pass


def _parse_argv(argv):
    logger.debug(f"argv\t{argv}")
    doc = f"""
    {__file__}
    """
    parser = argparse.ArgumentParser(doc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    now = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    parser.add_argument("--agent-seed", required=True, type=int, help="Seed for the agent.")
    parser.add_argument("--alpha", required=False, type=float, default=1e-3, help="α for TD error.")
    parser.add_argument("--dat-file", default=os.path.join("log", now + "_" + str(os.getpid()) + "_" + os.path.basename(__file__) + ".dat"), help="Set data file.")
    parser.add_argument("--dqn-mode", required=False, default="doubledqn", choices=["doubledqn", "dqn"], type=str, help="Type of DQN.")
    parser.add_argument("--env-seed", required=True, type=int, help="Seed to reset the environment.")
    parser.add_argument("--epsilon", required=True, type=float, help="ε-greedy.")
    parser.add_argument("--gamma", required=True, type=float, help="γ for discounted reward.")
    parser.add_argument("--log-file", default=os.path.join("log", now + "_" + str(os.getpid()) + "_" + os.path.basename(__file__) + ".log"), help="Set log file.")
    parser.add_argument("--log-file-level", default="debug", type=lambda x: getattr(logging, x.upper()), help="Set log level for the log file.")
    parser.add_argument("--log-stderr-level", default="info", type=lambda x: getattr(logging, x.upper()), help="Set log level for stderr.")
    parser.add_argument("--lr", required=True, type=float, help="Learning rate.")
    parser.add_argument("--n-batch", required=True, type=int, help="Batch size.")
    parser.add_argument("--n-episodes", required=True, type=int, help="Number of episodes to run.")
    parser.add_argument("--n-epsilon-decay", required=True, type=int, help="Number of steps to decay epsilon.")
    parser.add_argument("--n-log-steps", required=True, type=int, help="Record logs per this steps")
    parser.add_argument("--n-middle", required=True, type=int, help="Number of units in a hidden layer.")
    parser.add_argument("--n-replay-memory", required=True, type=int, help="Capacity of the replay memory.")
    parser.add_argument("--n-steps", required=True, type=int, help="Number of steps to run per episode.")
    parser.add_argument("--n-target-update-episodes", required=True, type=int, help="Number of episodes to update the target network.")
    parser.add_argument("--q-target-mode", required=False, default="mnih2015", type=str, choices=["mnih2015", "td"], help="Implicit vs explicit α.")
    parser.add_argument("--replay-memory-seed", required=True, type=int, help="Random state for minibatch.")
    parser.add_argument("--torch-seed", required=True, type=int, help="Seed for PyTorch.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)
    logger.debug(f"args\t{args}")
    assert args.n_batch <= args.n_replay_memory, (args.n_batch, args.n_replay_memory)
    assert 0 < args.alpha, args.alpha
    assert 0 < args.n_epsilon_decay
    return args


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
    logger.setLevel(logging.DEBUG)
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
