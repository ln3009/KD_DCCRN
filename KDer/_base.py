# -*- coding: utf-8 -*-
"""
KDer/_base.py
最基础的蒸馏基类：
- 载入 student / teacher
- 初始化优化器、调度器、AMP Scaler
- 统一的日志与可视化
- 保存/恢复训练（含蒸馏额外模块，如适配器）
- 提供 STFT/ISTFT 与默认任务损失（负 SI-SDR）
- 训练/验证循环（子类只需实现 compute_losses）
"""


import sys
import os
import argparse
from typing import Dict, Tuple, Iterable, List, Optional

import toml
from tqdm import tqdm

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ====== Pillow 兼容性补丁 ======
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
# ===========================================================

# 项目根目录入 sys.path（与现有代码保持一致）
sys.path.append(os.getcwd())

# 复用项目里的工具/度量
# 数据集类
from dataset.dataset import DNS_Dataset
# 教师模型类
from module.dc_crn import DCCRN

from dataset.compute_metrics import compute_metric
from audio.utils import prepare_empty_path, print_networks
from audio.metrics import SI_SDR, transform_pesq_range
from audio.feature import EPS

plt.switch_backend("agg")


class Distiller(nn.Module):
    """
    通用蒸馏基类：
    - 提供与 BaseTrainer 对齐的训练基础设施
    - teacher 的加载/冻结
    - 可注册蒸馏的“额外可学习模块”（如 1x1 conv 适配器）并参与优化与保存
    - 训练/验证循环
    - 子类必须实现：compute_losses(noisy, clean) -> Dict[str, Tensor]
      返回的字典至少包含：loss_total、loss_sup（监督/任务损失）、loss_kd（蒸馏损失）
    """
    