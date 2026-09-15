"""Graph WaveNet backend shared by the STD-MAE and STEP baselines.

Both papers ship a near-identical fork of Wu et al. (2019)'s Graph WaveNet,
differing only in how the pre-trained representation is injected into the skip
path — STEP has one stream (TSFormer), STD-MAE has two (T-MAE and S-MAE). That
is the ``hidden_streams`` argument here; everything else is the reference model.

Keeping ONE copy matters for the table: if the two arms drifted apart in the
backend, a gap between their rows could come from the forecaster rather than
from the pre-training objective, which is the thing being compared.

Changes from both references, all mechanical:

  * ``fc_his``'s ``96`` and the hard-coded ``hidden_states`` split are driven by
    ``hidden_dim`` / ``hidden_streams``, so the backend follows the encoder
    width instead of assuming METR-LA's.
  * ``forward(..., supports=...)`` takes the graph per call. STEP always does
    this (it learns the graph); STD-MAE does it on a dynamic-topology dataset
    and otherwise uses the static buffer registered at construction. ``nconv``'s
    einsum already accepted a batched ``(B, N, N)`` adjacency.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class nconv(nn.Module):
    def forward(self, x, A):
        A = A.to(x.device)
        if A.dim() == 3:                                # per-sample adjacency
            return torch.einsum("ncvl,nvw->ncwl", (x, A)).contiguous()
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0),
                             stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """Graph WaveNet with a pre-trained representation added to the skip path.

    ``hidden_streams``: how many ``hidden_dim``-wide blocks ``hidden_states``
    carries. 1 = STEP (TSFormer). 2 = STD-MAE (T-MAE ++ S-MAE), split in half
    and each given its own projection, as in that paper.
    """

    def __init__(self, num_nodes, supports=None, dropout=0.3, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2,
                 hidden_dim=96, hidden_streams=1, num_supports=0, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_streams = hidden_streams

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.fc_his = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, end_channels), nn.ReLU(),
                          nn.Linear(end_channels, skip_channels), nn.ReLU())
            for _ in range(hidden_streams)])
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels, kernel_size=(1, 1))

        # A static graph is a buffer so `.to(device)` moves it; a per-batch or
        # learned one is handed in per forward, and only its COUNT is needed
        # here (it sizes the gcn input width, fixed at construction).
        if supports is not None:
            stacked = torch.stack([torch.as_tensor(s, dtype=torch.float32)
                                   for s in supports])
            self.register_buffer("static_supports", stacked)         # (S, N, N)
            self.supports_len = stacked.shape[0]
        else:
            self.register_buffer("static_supports", None)
            self.supports_len = int(num_supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))
            else:
                m, p, n = torch.svd(aptinit)
                self.nodevec1 = nn.Parameter(
                    torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5)))
                self.nodevec2 = nn.Parameter(
                    torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t()))
            self.supports_len += 1

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(
                    residual_channels, dilation_channels,
                    kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(
                    residual_channels, dilation_channels,
                    kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(
                    dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(
                    dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels,
                                          dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels,
                                    kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim,
                                    kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input: torch.Tensor, hidden_states: torch.Tensor,
                supports: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """``input``: [B, L, N, C]. ``hidden_states``: [B, N, hidden_streams*d].
        ``supports``: optional graph, each entry [B, N, N] or [N, N].
        Returns [B, N, out_dim]."""
        if supports is None and self.static_supports is not None:
            supports = list(self.static_supports)
        elif supports is None:
            supports = []
        else:
            supports = list(supports)

        input = input.transpose(1, 3)                       # B, C, N, L
        input = F.pad(input, (1, 0, 0, 0))
        input = input[:, :self.in_dim, :, :]
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = supports
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except TypeError:                               # first block, skip == 0
                skip = 0
            skip = s + skip

            if self.gcn_bool and new_supports:
                x = self.gconv[i](x, new_supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        d = self.hidden_dim
        for j, fc in enumerate(self.fc_his):
            h = fc(hidden_states[:, :, j * d:(j + 1) * d])
            skip = skip + h.transpose(1, 2).unsqueeze(-1)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.squeeze(-1).transpose(1, 2)                # B, N, out_dim
