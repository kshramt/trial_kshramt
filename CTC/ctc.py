#!/usr/bin/python3

import logging

import torch


__version__ = "0.1.0"
logger = logging.getLogger(__name__)


BLANK = 0


class CTCLoss1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pred, target, blank=BLANK):
        """
        CTC loss without mini-batch.

        `pred`:: Sequence x Class.
        `target`:: Sequence.
        """

        n_seq_pred, n_class = pred.shape
        n_seq_target, = target.shape
        assert n_seq_target <= n_seq_pred, (n_seq_target, n_seq_pred)
        path = _path_of(target, blank)
        n_path, = path.shape

        col_prev = torch.empty(n_path + 2)  # path length + padding at the top for transition
        col_prev.fill_(float("-inf"))
        col_prev[2] = pred[0, 0]
        col_prev[3] = pred[0, path[0]]

        for i in range(1, n_seq_pred):
            col_now = _transition(i, col_prev, pred, target)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad, = grad_outputs
        return ret, None, None, None


def _transition(i, col_prev, pred, target):
    ll = pred[i, target]
    col_now = torch.empty_like()


def _logsumexp(xs):
    xmax = xs.max()
    return xmax + (xs - xmax).exp().sum().log()


def _path_of(target, blank):
    n_seq, = target.shape
    path = torch.empty(size=(2*n_seq + 1,), dtype=target.dtype, layout=target.layout, device=target.device)
    path.fill_(blank)
    path[1::2, :] = target
    return path


def beam_search():
    pass
