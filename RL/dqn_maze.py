#!/usr/bin/python

import argparse
import collections
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


tq = np.array([
    [
        [-0.4313128235610175, 0.0, 0.0, -0.4312855316641398, ],
        [-0.40139519340735025, 0.0, -0.9999999999998365, 0.5986742904083278, ],
        [-0.9999999999999069, 0.0, -0.4313231566563678, 0.6301807386843181, ],
        [-0.09750219030213195, 0.0, -0.4015417517927016, -0.0975004457539218, ],
        [-0.05075849601338359, 0.0, -0.9999998878588902, 0.9499992607310138, ],
        [-0.9999203577771005, 0.0, -0.09777920040581166, 0.9995672227742406, ],
        [0.0, 0.0, -0.052752784484314946, -0.05049146584804887, ],
    ],
    [
        [0.5986885066278026, -0.9999999999999912, 0.0, -0.401298906872775, ],
        [0.6302048818172881, -0.4313075069243088, -0.4312877363881571, 0.6302190072751246, ],
        [-0.09750033431994773, -0.40139897868356755, 0.5986651635325801, 0.6634003857251853, ],
        [0.9499998312779898, -0.9999999999884579, 0.6301979293185477, -0.14262562392987627, ],
        [0.9999999086603651, -0.09750435799158161, -0.09750083496981937, 0.9024988179642065, ],
        [0.0, 0.0, 0.0, 0.0, ],
        [0.0, -0.9999569645214524, 0.9999117537696405, -0.09796125918977232, ],
    ],
    [
        [0.630230422337953, -0.4312924974872748, 0.0, -0.3697708158163603, ],
        [0.6634040035876616, 0.5986885454177628, -0.4012979304394517, 0.6634097280681747, ],
        [-0.14262550639313232, 0.6302055622632289, 0.6302297394754544, 0.6983283215975835, ],
        [0.9024997474969813, -0.09750026462692407, 0.6634012217965989, -0.18549513232747203, ],
        [0.9499999054117878, 0.9499997736742095, -0.14262566387530795, 0.8573734252022962, ],
        [-0.09751389626206793, 0.9999999848300043, 0.9024987004393302, 0.9024963311935685, ],
        [0.0, -0.050190664296804334, 0.9499985128574511, -0.14263180590794008, ],
    ],
    [
        [0.663409168205478, -0.4012978831457621, 0.0, -0.33659141667791237, ],
        [0.6983306036548723, 0.630231650816184, -0.36977073352773454, 0.6983313461788969, ],
        [-0.18549497324790765, 0.6634038381703173, 0.6634101038584873, 0.7350884764133461, ],
        [0.8573743512921884, -0.14262546177599825, 0.698329808386798, 0.7737791427286396, ],
        [0.9024996948000832, 0.9024996578307266, -0.1854951101502572, 0.8145049577212665, ],
        [-0.14262719643254973, 0.9499999264619926, 0.8573740149966778, 0.8573742488263986, ],
        [0.0, -0.09750337830424105, 0.9024995022837863, -0.18549822602890684, ],
    ],
    [
        [0.6983295719453971, -0.3697698717983025, 0.0, -0.3697750886007327, ],
        [0.7350888362642282, 0.6634105177454633, -0.336593268237506, 0.6634106938473315, ],
        [0.7737793912739986, 0.6983304758962137, 0.6983315929780308, 0.6983298857709268, ],
        [0.8145054134570691, -0.18549473092866473, 0.7350885037026882, -0.26491078959891284, ],
        [0.8573745275004758, 0.8573744836759396, 0.7737791836250534, 0.7737789019074022, ],
        [-0.18549495453644474, 0.9024998004939888, 0.8145052262144978, 0.8145050075746595, ],
        [0.0, -0.1426266113677714, 0.8573745075287679, -0.22622122353317464, ],
    ],
    [
        [0.6634059468559953, -0.3365953580771118, 0.0, -0.9999999999999989, ],
        [0.6983310315213149, 0.6983317680579202, -0.36978142621266447, -0.36976705123945, ],
        [-0.26491101643167525, 0.7350883700416386, 0.6634097492762576, -0.336589898472616, ],
        [0.7737793152666022, 0.7737793269262768, 0.6983295830656842, -0.9999999999999989, ],
        [0.8145053246230859, 0.8145054254113102, -0.26491119428103616, -0.2649111212724126, ],
        [-0.22622218008392897, 0.8573745402638185, 0.7737794597942503, -0.22622166106441027, ],
        [0.0, -0.18549478102641018, 0.81450524891344, -0.9999999999999767, ],
    ],
    [
        [-0.36976705554140826, -0.36978164949442194, 0.0, 0.0, ],
        [-0.3365895475652871, 0.6634108818333803, -0.9999999999999989, 0.0, ],
        [-0.9999999999999989, 0.6983306693933249, -0.369766790453253, 0.0, ],
        [-0.26491146754489936, -0.264912151548666, -0.33659057877494036, 0.0, ],
        [-0.22622140768364704, 0.7737793477938228, -0.9999999999999989, 0.0, ],
        [-0.9999999999999989, 0.8145051815430273, -0.2649127807055377, 0.0, ],
        [0.0, -0.2262214663184031, -0.22622232226440842, 0.0, ],
    ],
], dtype=float)
maze = np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0, -1,  0,  1, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1,  0,  0,  0,  0,  0, -1],
    [-1,  0,  0, -1,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1],
], dtype=float)
# maze = np.array([
#     [-1, -1, -1, -1, -1, -1],
#     [-1,  0,  0, -1,  1, -1],
#     [-1,  0,  0, -1,  0, -1],
#     [-1,  0,  0,  0,  0, -1],
#     [-1,  0,  0, -1,  0, -1],
#     [-1, -1, -1, -1, -1, -1],
# ], dtype=float)
# maze = np.array([
#     [-1, -1, -1, -1, -1, -1],
#     [-1,  0,  0, -1,  1, -1],
#     [-1,  0,  0,  0,  0, -1],
#     [-1, -1, -1, -1, -1, -1],
# ], dtype=float)


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
        loss = torch.nn.MSELoss
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

    def train(self, si, ai1, ri1, si1, done):
        self.replay_memory.push((self.prep_s(si), ai1, ri1, self.prep_s(si1), done))
        if self.replay_memory.filled():
            return self.optimize()

    def optimize(self):
        batch = Transition(*zip(*self.replay_memory.sample(self.n_batch)))
        # logger.debug(self.replay_memory.buffer)
        # logger.debug(batch)
        # exit(1)

        # Instead of adding Q(terminal state, any action) = 0 in training data,
        # we directly use that fact to reduce approximation errors.
        self.model.eval()
        # for done, si, ai1, ri1 in zip(batch.done, batch.si, batch.ai1, batch.ri1):
        #     si = (int(round(si[0]*maze.shape[0])), int(round(si[1]*maze.shape[1])))
        #     if done:
        #         logger.debug(f"done\t{si}\t{ai1}\t{ri1}")
        v_hat_si1 = var(torch.zeros(self.n_batch), volatile=True)
        non_final_si1 = [x for x, done in zip(batch.si1, batch.done) if not done]
        # logger.info(batch.si1)
        if non_final_si1:
            mask = self._byte_tensor([not x for x in batch.done])
            v_hat_si1[mask] = self.model(var(self._float_tensor(non_final_si1), volatile=True)).max(1)[0]
            # logger.info(f"v_hat_si1[mask == 0]\t{v_hat_si1[mask == 0]}")
        v_hat_si1.volatile = False # used as a constant later
        # logger.debug(f"v_hat_si1\t{v_hat_si1}")
        # q_bellman = (var(self._float_tensor(batch.ri1)) + self.gamma*v_hat_si1).view(self.n_batch, -1)
        inds = np.array(batch.si)
        inds[:, 0] *= maze.shape[0]
        inds[:, 1] *= maze.shape[1]
        inds = np.round(inds).astype(int)
        q_bellman = var(self._float_tensor(tq[inds[:, 0], inds[:, 1], np.array(batch.ai1, dtype=int)]).view(-1, 1))
        # if random.random() < 0.01:
        #     logger.debug(q_bellman)

        self.model.train()
        q_pred = self.model(var(self._float_tensor(batch.si))).gather(1, var(self._long_tensor(batch.ai1).view(-1, 1)))
        # if random.random() < 0.05:
        #     logger.debug(torch.cat([q_bellman, q_pred], dim=1))
        # exit(1)

        # np.set_printoptions(precision=2)
        # logger.info(f"q_bellman.view(-1)[mask == 0]\t{q_bellman.view(-1)[mask == 0]}")
        # logger.info(f"q_bellman[(mask == 0).view(-1, 1)]\t{q_bellman[(mask == 0).view(-1, 1)]}")
        # logger.debug(f"q_bellman.shape\t{q_bellman.shape}")
        # logger.info(f"np.array(batch.ri1)[batch.done]\t{np.array(batch.ri1)[np.array(batch.done)]}")
        # logger.info(f"v_hat_si1[mask == 0]\t{v_hat_si1[mask == 0]}")
        # logger.debug(f"q_bellman[mask == 0]\t{q_bellman[mask == 0]}")
        # q_pred_const = var(q_pred.data)
        # td = q_bellman - q_pred_const
        # logger.debug(f"td\t{td}")
        # import time
        # time.sleep(0.5)
        # print(q_bellman.view(1, -1).data.numpy())
        # print(q_pred.view(1, -1).data.numpy())
        # print()

        # q_target = q_pred_const + self.alpha*td
        q_target = q_bellman # Mnih et al (2015, Nature)
        # logger.debug(self.model(var(self._float_tensor([[1, 1]]))).data.numpy())
        # for i in range(self.n_batch):
        #     if batch.done[i]:
        #         logger.debug(f"qq\t{i}\t{q_pred[i, 0].data.numpy()[0]}\t{q_target[i, 0].data.numpy()[0]}")
        loss = self.loss(q_pred, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        q_pred = self.model(var(self._float_tensor(batch.si), volatile=True)).gather(1, var(self._long_tensor(batch.ai1).view(self.n_batch, -1)))
        # for i in range(self.n_batch):
        #     if batch.done[i]:
        #         if random.random() < 1/12:
        #             logger.debug(f"qq\t{i}\t{q_pred[i, 0].data.numpy()[0]}\t{q_target[i, 0].data.numpy()[0]}")
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
        # logger.debug((self.state, ri1))
        # logger.debug(f"env\t{(self.state, ri1, bool(ri1 == 1))}")
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
    # act = Log1p
    # act = torch.nn.Tanh
    model = torch.nn.Sequential(collections.OrderedDict((
        (namer("fc"), torch.nn.Linear(n_input, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        (namer("fc"), torch.nn.Linear(args.n_middle, args.n_middle)),
        (namer("ac"), act()),
        ("output", torch.nn.Linear(args.n_middle, n_output)),
    )))
    def _init(m):
        if type(m) == torch.nn.Linear:
           torch.nn.init.kaiming_uniform(m.weight.data)
           # torch.nn.init.xavier_uniform(m.weight.data)
           m.bias.data.fill_(0)
    model.apply(_init)
    # opt = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=1e-1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            # if done:
            #     logger.debug(f"done\t{ri1}\t{si}\t{ai1}")
            metric = agent.train(si, ai1, ri1, si1, done)
            if i_step%args.n_log_steps == 0:
                step_result_list.append(dict(i_step=i_step, si=si, ai1=ai1, ri1=ri1, si1=si1, metric=metric))
                if metric is not None:
                    logger.info(f"loss\t{metric['loss'].data.numpy()[0]}")
            si = si1
            if done:
                episode_result_list.append(dict(i_episode=i_episode, env_seed=env_seed, step_result_list=step_result_list))
                break
        if i_episode%5 == 0:
            pass
            print_q(agent, *maze.shape)


def print_q(agent, n, m, fp=sys.stdout):
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

    # hdl = logging.StreamHandler(sys.stderr)
    hdl = logging.StreamHandler(sys.stdout)
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
