#!/usr/bin/python3

import logging

import torch


__version__ = "0.1.0"
logger = logging.getLogger(__name__)


BLANK = -1

class CTCLoss(torch.nn.Module):
    """
    `y_pred`:: Input sequence x Batch x Class.
    `y_true`:: Output sequence x Batch
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        n_in_seq, n_batch, n_class = y_pred.shape
        n_out_seq, n_out_batch = y_true.shape
        assert n_out_seq <= n_in_seq, (n_out_seq, n_in_seq)
        assert n_out_batch == n_batch, (n_out_batch, n_batch)
        y_true_ctc = y_ctc_of(y_true, BLANK)


def y_ctc_of(y, blank=BLANK):
    n_seq, n_batch = y.shape
    y_ctc = torch.empty(size=(2*n_seq + 1, n_batch), dtype=y.dtype, layout=y.layout, device=y.device)
    y_ctc.fill_(blank)
    y_ctc[1::2, :] = y
    return y_ctc


def beam_search():
    pass
