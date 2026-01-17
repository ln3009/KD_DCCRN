# -*- coding: utf-8 -*-
# KD_FitNets（带子空间能量占比监控）

import sys
import os
import argparse
import toml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from audio.utils import prepare_empty_path, print_networks


class FitNets_Trainer(BaseTrainer):
    """
    FitNets-style Knowledge Distillation：
      Loss = (1 - v) * student_supervised_loss + v * feature_distillation_loss

    其中：
      feature_distillation_loss = mean_i MSE( adapter_i(s_feat_i), t_feat_i )

    本版本新增“子空间能量占比”监控：
      给定教师第 l 层协方差 Ct，取其 top-k 特征向量 Uk，定义 Pk = Uk Uk^T，
      记录 ratio_k = tr(Pk * Cs) / tr(Cs)，其中 Cs 为学生映射后特征协方差。
    """

    def __init__(self, config, teacher_model, model, train_iter, valid_iter, device="cpu"):
        #  the config is config of student
        super().__init__(config, model, train_iter, valid_iter, device=device)

        # get student model
        self.model.to(self.device)

        # get teacher model
        self.teacher_model = teacher_model

        # get knowledge_distillation args
        self.teacher_model_path = config["knowledge_distillation"]["teacher_model"]
        self.v = config["knowledge_distillation"]["v"]

        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "fitnets_x1_16demo3")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "fitnets_x1_16demo3")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # to device
        self.teacher_model = self.teacher_model.to(self.device)
        # load teacher model
        self.load_teacher_model()
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        # ====== 构建 1x1 conv 适配器以对齐 student/teacher 每层通道 ======
        # 用 dummy 数据触发一次前向，拿到各层特征形状
        n_fft = config["dataset"]["n_fft"]
        Fbins = n_fft // 2 + 1
        Tbins = 100
        dummy = torch.randn(1, Fbins, Tbins, 2, device=self.device)

        with torch.no_grad():
            _, t_feats = self.teacher_model(dummy, return_features=True)
            _, s_feats = self.model(dummy,   return_features=True)

        # 为每一层特征创建一个 1×1 Conv 适配器：C_s -> C_t
        self.adapters = nn.ModuleList()
        for fs, ft in zip(s_feats, t_feats):
            C_s = fs.shape[1]
            C_t = ft.shape[1]
            self.adapters.append(nn.Conv2d(C_s, C_t, kernel_size=1))
        for adapter in self.adapters:
            adapter.to(self.device)

        # 将 adapters 的参数加入 optimizer
        self.optimizer.add_param_group({"params": self.adapters.parameters()})

        # 同步 scheduler 的 lrs 列表长度
        if hasattr(self.scheduler, "base_lrs"):
            new_lr = self.optimizer.param_groups[-1]['lr']
            self.scheduler.base_lrs.append(new_lr)
        if hasattr(self.scheduler, "min_lrs"):
            self.scheduler.min_lrs.append(self.scheduler.min_lrs[0])

        print_networks(list(self.adapters))

        # ====== 子空间监控配置（可由 config 覆盖） ======
        kd_cfg = config.get("knowledge_distillation", {})
        mon_cfg = kd_cfg.get("subspace_monitor", {})
        self.monitor_enabled   = bool(mon_cfg.get("enabled", True))
        self.monitor_layer_idx = int(mon_cfg.get("layer_idx", 1))      # 监控第几层（默认 0）
        self.monitor_k_list    = list(mon_cfg.get("k_list", [8, 16, 32]))
        self.monitor_centered  = bool(mon_cfg.get("centered", True))   # 协方差是否中心化
        self.monitor_freq      = int(mon_cfg.get("freq", 1))           # 每多少个 epoch 记录一次

    # ----------------------- 工具函数：子空间监控 -----------------------
    @staticmethod
    def _flatten_feat(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, F, T] or [B, C, H, W]
        return: [C, N] where N = B*F*T
        """
        assert x.dim() == 4, f"Expect 4D feature, got {tuple(x.shape)}"
        return x.permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)

    @staticmethod
    def _cov_from_feat(mat: torch.Tensor, centered: bool = True) -> torch.Tensor:
        """
        mat: [C, N] -> covariance-like Gram: [C, C]
        中心化后计算 (X X^T)/N；若不中心化，使用原值。
        """
        if centered:
            mat = mat - mat.mean(dim=1, keepdim=True)
        N = mat.size(1)
        return (mat @ mat.transpose(0, 1)) / max(N, 1)

    @torch.no_grad()
    def _subspace_energy_ratio(self, fs: torch.Tensor, ft: torch.Tensor, adapter: nn.Module,
                               k_list, centered=True):
        """
        计算并返回两个 dict：
          ratios_student_in_teacher[k] = tr(P_k * C_s) / tr(C_s)
          teacher_energy_frac[k]       = sum_{i=1}^k lambda_i / sum_i lambda_i
        其中 P_k = U_k U_k^T, U_k 为 Ct 的前 k 个特征向量。
        """
        # 学生映射到教师通道维
        fs_proj = adapter(fs)                     # [B, C_t, H, W]
        Xs = self._flatten_feat(fs_proj).float()  # [C_t, N]
        Xt = self._flatten_feat(ft).float()       # [C_t, N]

        Cs = self._cov_from_feat(Xs, centered=centered)  # [C_t, C_t]
        Ct = self._cov_from_feat(Xt, centered=centered)  # [C_t, C_t]

        # 对称阵特征分解（升序），翻转为降序
        eigvals, eigvecs = torch.linalg.eigh(Ct)         # eigvecs: [C_t, C_t]
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)

        tr_Cs = torch.trace(Cs).clamp_min(1e-12)
        tr_Ct = torch.trace(Ct).clamp_min(1e-12)

        ratios_student_in_teacher = {}
        teacher_energy_frac = {}

        C = Cs.shape[0]
        for k in k_list:
            kk = int(min(max(1, int(k)), C))
            Uk = eigvecs[:, :kk]           # [C_t, k]
            # energy_in = tr(Uk^T Cs Uk) = sum_i u_i^T Cs u_i
            CsUk = Cs @ Uk                 # [C_t, k]
            energy_in = (Uk * CsUk).sum()  # 标量
            ratios_student_in_teacher[kk] = (energy_in / tr_Cs).item()

            # 教师的对照：前 k 特征值之和 / 总能量
            teacher_energy_frac[kk] = (eigvals[:kk].sum() / tr_Ct).item()

        return ratios_student_in_teacher, teacher_energy_frac
    # ------------------------------------------------------------------

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path), f"Teacher model not found: {self.teacher_model_path}"
        load_model = torch.load(self.teacher_model_path, map_location="cpu")
        self.teacher_model.load_state_dict(load_model)
        print(f"Loaded teacher weights from {self.teacher_model_path} done")

    def feature_distillation_loss(self, s_feats, t_feats):
        """
        对 student 特征依次做 1×1 conv 映射，再与教师特征计算 MSE，并求均值
        """
        loss = 0.0
        for adapter, fs, ft in zip(self.adapters, s_feats, t_feats):
            fs_proj = adapter(fs)
            loss += F.mse_loss(fs_proj, ft)
        return loss / len(s_feats)

    def save_checkpoint(self, epoch, is_best_epoch=False):
        # 先调用 BaseTrainer 的保存逻辑（会写 latest/best model & checkpoint）
        super().save_checkpoint(epoch, is_best_epoch)

        # 再额外保存 adapters 的权重
        prefix = "best" if is_best_epoch else "latest"
        adapter_path = os.path.join(self.checkpoints_path, prefix + "_adapters.pth")
        torch.save(self.adapters.state_dict(), adapter_path)
        print(f"Saved adapters state")

    def resume_checkpoint(self):
        # 先调用 BaseTrainer 的恢复逻辑（会加载 latest checkpoint.tar 里的 model、optimizer、scheduler、scaler）
        super().resume_checkpoint()

        # 再尝试加载 adapters 权重
        adapter_path = os.path.join(self.checkpoints_path, "latest_adapters.pth")
        if os.path.exists(adapter_path):
            adapters_state = torch.load(adapter_path, map_location="cpu")
            self.adapters.load_state_dict(adapters_state)
            print(f"Loaded adapters state from {adapter_path}")
        else:
            print(f"No adapter checkpoint found at {adapter_path}, skipping adapter load.")

    def train_epoch(self, epoch):
        loss_total = 0.0
        student_loss_total = 0.0
        feat_loss_total = 0.0

        for batch_idx, (noisy, clean) in enumerate(tqdm(self.train_iter, desc="Train FitNets")):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            with torch.no_grad():
                t_mask, t_feats = self.teacher_model(noisy_spec, return_features=True)  # [B, 2, F, T]

            with autocast(enabled=self.use_amp):
                s_mask, s_feats = self.model(noisy_spec, return_features=True)          # [B, 2, F, T]

            # ===== 子空间监控：每 epoch 的第一个 batch，按 freq 控制频率 =====
            if (self.monitor_enabled
                and (epoch % max(self.monitor_freq, 1) == 0)
                and batch_idx == 0):
                li = min(self.monitor_layer_idx, len(s_feats) - 1, len(t_feats) - 1)
                try:
                    ratios, t_upper = self._subspace_energy_ratio(
                        fs=s_feats[li], ft=t_feats[li],
                        adapter=self.adapters[li],
                        k_list=self.monitor_k_list,
                        centered=self.monitor_centered
                    )
                    msg = [f"k={k}: ratio={ratios[k]:.4f} | teacher_frac={t_upper[k]:.4f}"
                           for k in sorted(ratios.keys())]
                    print(f"[Epoch {epoch}] Subspace@layer{li} -> " + " ; ".join(msg))
                    for k, v in ratios.items():
                        self.writer.add_scalar(f"subspace/layer{li}_student_in_teacher_k{k}", v, epoch)
                    for k, v in t_upper.items():
                        self.writer.add_scalar(f"subspace/layer{li}_teacher_energy_frac_k{k}", v, epoch)
                except Exception as e:
                    print(f"[Epoch {epoch}] Subspace monitor failed: {e}")

            # ===== 常规前向与损失 =====
            student_enh = self.audio_istft(s_mask, noisy_spec)
            student_loss = self.loss(student_enh, clean)
            feat_loss = self.feature_distillation_loss(s_feats, t_feats)
            loss = (1 - self.v) * student_loss + self.v * feat_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            all_params = list(self.model.parameters()) + list(self.adapters.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            student_loss_total += float(student_loss.item())
            feat_loss_total += float(feat_loss.item())
            loss_total += float(loss.item())

        avg_student = student_loss_total / max(len(self.train_iter), 1)
        avg_feat = feat_loss_total / max(len(self.train_iter), 1)
        avg_loss = loss_total / max(len(self.train_iter), 1)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Sup: {avg_student:.4f} | Feat: {avg_feat:.4f}")
        self.update_scheduler(avg_loss)

        self.writer.add_scalar("loss/student", avg_student, epoch)
        self.writer.add_scalar("loss/feature", avg_feat, epoch)
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)


if __name__ == "__main__":
    '''
    运行示例：
    python trainer/KD_FitNets_trainerdemocopy.py -TC config/base_config.toml -SC config/KD_FitNets_config.toml --gpu 3

    TensorBoard：
    tensorboard --logdir logs/train/fitnets_x1_16demo1
    '''
    parser = argparse.ArgumentParser(description="FitNets Knowledge Distillation Trainer")
    parser.add_argument("-TC", "--teacher_config", required=True, type=str, help="Teacher Config (*.toml).")
    parser.add_argument("-SC", "--student_config", required=True, type=str, help="Student Config (*.toml).")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="指定要使用的 GPU 编号 (例如 0, 1, 2, …)，如果只有 CPU 可用则忽略。"
    )
    args = parser.parse_args()

    # config device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    # get teacher config
    config_t = toml.load(args.teacher_config)
    teacher_model = globals().get(config_t["model"]["name"])(
        n_fft=config_t["dataset"]["n_fft"],
        rnn_layers=config_t["model"]["rnn_layers"],
        rnn_units=config_t["model"]["rnn_units"],
        kernel_num=config_t["model"]["kernel_num"],
        kernel_size=config_t["model"]["kernel_size"],
    )

    # get student config
    config = toml.load(args.student_config)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")
    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config student model
    model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # 初始化 Trainer 并开始训练
    trainer = FitNets_Trainer(config, teacher_model, model, train_iter, valid_iter, device)
    trainer()
