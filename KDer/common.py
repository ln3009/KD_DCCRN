# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvReg(nn.Module):
    """
    1×1 卷积（对齐通道数用），支持 [B, C, F, T] 的特征映射。
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def grab_feat_shapes_via_dummy(
    student: nn.Module,
    teacher: nn.Module,
    n_fft: int,
    device: torch.device,
    t_bins: int = 100,
) -> Tuple[List[torch.Size], List[torch.Size]]:
    """
    给定 n_fft 构造一个 dummy 频谱，跑一遍 student/teacher，拿到特征 shapes。
    约定模型 forward(noisy_spec, return_features=True) -> (primary_output, List[feat])
    其中 noisy_spec 形状为 [B, F, T, 2]。
    """
    f_bins = n_fft // 2 + 1
    dummy = torch.randn(1, f_bins, t_bins, 2, device=device)
    with torch.no_grad():
        _, t_feats = teacher(dummy, return_features=True)
        _, s_feats = student(dummy, return_features=True)
    s_shapes = [f.shape for f in s_feats]
    t_shapes = [f.shape for f in t_feats]
    return s_shapes, t_shapes


def build_channel_adapters(
    s_shapes: List[torch.Size],
    t_shapes: List[torch.Size],
    device: torch.device,
) -> nn.ModuleList:
    """
    针对每层特征，用 1×1 Conv 把 student 通道数映射到 teacher 通道数。
    输入输出特征统一是 [B, C, F, T]。
    """
    adapters = nn.ModuleList()
    for s, t in zip(s_shapes, t_shapes):
        c_s = s[1]
        c_t = t[1]
        adapters.append(ConvReg(c_s, c_t).to(device))
    return adapters
