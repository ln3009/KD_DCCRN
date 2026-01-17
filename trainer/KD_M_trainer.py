# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.append(os.getcwd())
from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from audio.utils import prepare_empty_path

class KD_M_trainer(BaseTrainer):
    """
    基于边际差的响应蒸馏（pairwise margin / ranking）
    总损失：
        Loss = (1 - v) * L_task + v * L_marginKD

    其中 L_marginKD 在每一帧的频带内，基于教师的 log-magnitude 选择 Top-K 频带构造成对约束：
        hinge( m - sign(Δ_t) * Δ_s )，只对 |Δ_t| >= δ 的“明确对”计入。
    """
    def __init__(self, config, teacher_model, model, train_iter, valid_iter, device="cpu"):
        super().__init__(config, model, train_iter, valid_iter, device=device)
        # get teacher model
        self.teacher_model = teacher_model
        self.teacher_model_path = config["knowledge_distillation"]["teacher_model"]
        
        # KD 超参
        kd_cfg = config.get("knowledge_distillation", {})
        self.v = float(kd_cfg.get("v", 0.3))                        # KD 权重
        self.margin_m = float(kd_cfg.get("margin", 0.1))            # 排名间隔 m
        self.delta = float(kd_cfg.get("delta", 0.0))                # 模糊阈值 δ（忽略 |Δ_t| < δ 的对）
        self.topk = int(kd_cfg.get("topk", 64))                     # 每帧 Top-K 频带
        self.use_log_spectrum = bool(kd_cfg.get("use_log_spec", True))  # 是否用 估计干净谱 的 log-mag 作为响应
        

        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "KD_R_margin_pair_x1_16_2")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "KD_R_margin_pair_x1_16_2")
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # to device
        self.teacher_model = self.teacher_model.to(self.device)
        # load teacher model
        self.load_teacher_model()
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        
        self._eps = 1e-8

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path)
        load_obj = torch.load(self.teacher_model_path, map_location="cpu")
        if isinstance(load_obj, dict):
            if "state_dict" in load_obj:
                state_dict = load_obj["state_dict"]
            elif "model" in load_obj:
                state_dict = load_obj["model"]
            else:
                state_dict = load_obj
        else:
            state_dict = load_obj
        self.teacher_model.load_state_dict(state_dict)
        print("Load teacher model done...")

    # ======== 工具：统一把 [B,F,T,2] 或 [B,2,F,T] 规格转成 [B,2,F,T] ========
    @staticmethod
    def _ensure_b2ft(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expect 4D tensor, got {x.shape}")
        if x.size(1) == 2:
            return x  # [B,2,F,T]
        if x.size(-1) == 2:
            return x.permute(0, 3, 1, 2).contiguous()  # [B,F,T,2] -> [B,2,F,T]
        raise ValueError(f"Unknown spec layout: {x.shape}")
    
    # ======== 工具：由 cRM 与噪声谱得到 估计干净谱 的 log-magnitude，形状 [B,F,T] ========
    def _mask_to_est_logmag(self, mask_b2ft: torch.Tensor, noisy_spec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # 统一规格
        mask = self._ensure_b2ft(mask_b2ft)                # [B,2,F,T]
        noisy = self._ensure_b2ft(noisy_spec)              # [B,2,F,T]
        mr, mi = mask[:, 0], mask[:, 1]                    # [B,F,T]
        xr, xi = noisy[:, 0], noisy[:, 1]
        # 复数乘法 S_hat = M ⊙ X
        Sr = mr * xr - mi * xi
        Si = mr * xi + mi * xr
        mag = torch.sqrt(Sr * Sr + Si * Si + eps)          # [B,F,T]
        return torch.log(mag + eps) if self.use_log_spectrum else mag
    
    # ======== 核心：边际差（pairwise margin / ranking）KD ========
    def response_margin_kd_loss(
        self,
        student_mask_b2ft: torch.Tensor,
        teacher_mask_b2ft: torch.Tensor,
        noisy_spec: torch.Tensor,
        m: float,
        delta: float,
        topk: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        在每一帧的频带内选择 Top-K（来自教师的 log-mag），构造 pairwise 排名铰链损失：
            relu( m - sign(Δ_t) * Δ_s )
        仅对 |Δ_t| >= δ 的成对关系计入。
        """
        # 计算师生“响应标量”：推荐用 估计干净谱 的 (对数)幅度
        t_log = self._mask_to_est_logmag(teacher_mask_b2ft, noisy_spec, eps=eps).detach()  # [B,F,T]
        s_log = self._mask_to_est_logmag(student_mask_b2ft, noisy_spec, eps=eps)          # [B,F,T]

        B, F, T = t_log.shape
        K = min(int(topk), F)
        # [B,T,F] 便于按帧 topk
        t_btF = t_log.transpose(1, 2).contiguous()   # [B,T,F]
        s_btF = s_log.transpose(1, 2).contiguous()   # [B,T,F]

        # 选每帧 Top-K（按教师响应）
        _, idx = torch.topk(t_btF, k=K, dim=2)       # [B,T,K]
        # gather 到 Top-K 子空间
        b_idx = idx.unsqueeze(-1).expand(-1, -1, -1, 1)  # [B,T,K,1]
        t_sel = torch.gather(t_btF, 2, idx)              # [B,T,K]
        s_sel = torch.gather(s_btF, 2, idx)              # [B,T,K]

        # 构造 pairwise 差：广播得到 [B,T,K,K]
        t_i = t_sel.unsqueeze(-1)
        t_j = t_sel.unsqueeze(-2)
        s_i = s_sel.unsqueeze(-1)
        s_j = s_sel.unsqueeze(-2)
        dT = t_i - t_j                                   # [B,T,K,K]
        dS = s_i - s_j

        # 掩蔽：忽略对角与教师差值过小的“模糊对”
        eye = torch.eye(K, device=dT.device, dtype=torch.bool).view(1, 1, K, K)
        mask_offdiag = ~eye
        mask_clear = (dT.abs() >= float(delta))
        pair_mask = (mask_offdiag & mask_clear)          # [B,T,K,K]

        # 铰链排名：relu( m - sign(dT) * dS )
        sign = torch.sign(dT)                             # [-1,0,1]
        hinge = torch.relu(float(m) - sign * dS)          # [B,T,K,K]

        # 只统计有效 pair
        if pair_mask.any():
            loss = (hinge * pair_mask.float()).sum() / (pair_mask.float().sum() + eps)
        else:
            loss = hinge.new_tensor(0.0)

        return loss

    def train_epoch(self, epoch):
        loss_total = 0.0
        kd_total = 0.0
        student_loss_total = 0.0

        for noisy, clean in tqdm(self.train_iter, desc="train"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            # 教师前向
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    teacher_mask = self.teacher_model(noisy_spec)  # 期望 [B,2,F,T] 或工程可兼容
                    teacher_mask = teacher_mask.detach()

        
            # 学生前向 + 损失
            with autocast(enabled=self.use_amp):
                student_mask = self.model(noisy_spec)

                # 任务损失
                student_enh = self.audio_istft(student_mask, noisy_spec)
                task_loss = self.loss(student_enh, clean)

                # 边际差 KD（pairwise ranking）
                kd_loss = self.response_margin_kd_loss(
                    student_mask_b2ft=student_mask,
                    teacher_mask_b2ft=teacher_mask,
                    noisy_spec=noisy_spec,
                    m=self.margin_m,
                    delta=self.delta,
                    topk=self.topk,
                    eps=self._eps,
                )

                loss = (1.0 - self.v) * task_loss + self.v * kd_loss

            # 反传优化
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()         


            # 统计
            student_loss_total += float(task_loss.item())
            kd_total += float(kd_loss.item())
            loss_total += float(loss.item())

        avg_student = student_loss_total / len(self.train_iter)
        avg_kd = kd_total / len(self.train_iter)
        avg_loss = loss_total / len(self.train_iter)

        print(f"[Epoch {epoch}]  Train Loss: {avg_loss:.4f} | Task: {avg_student:.4f} | KD(pair): {avg_kd:.4f}")
        self.update_scheduler(avg_student)

        self.writer.add_scalar("loss/student", avg_student, epoch)
        self.writer.add_scalar("loss/kd_pair", avg_kd, epoch)
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)


if __name__ == "__main__":
    '''
    python trainer/KD_M_trainer.py   -TC config/base_config.toml   -SC config/lite_v1_config.toml  --gpu 1
    '''
    parser = argparse.ArgumentParser(description="knowledge distillation trainer (pairwise margin KD)")
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

    # 教师配置与模型
    t_cfg = toml.load(args.teacher_config)
    teacher_model = globals().get(t_cfg["model"]["name"])(
        n_fft=t_cfg["dataset"]["n_fft"],
        rnn_layers=t_cfg["model"]["rnn_layers"],
        rnn_units=t_cfg["model"]["rnn_units"],
        kernel_num=t_cfg["model"]["kernel_num"],
        kernel_size=t_cfg["model"]["kernel_size"],
    )

    # 学生配置
    s_cfg = toml.load(args.student_config)

     # 数据集
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")
    batch_size = s_cfg["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else s_cfg["dataloader"]["num_workers"]
    drop_last = s_cfg["dataloader"]["drop_last"]
    pin_memory = s_cfg["dataloader"]["pin_memory"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, s_cfg, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, s_cfg, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config student model
    model = globals().get(s_cfg["model"]["name"])(
        n_fft=s_cfg["dataset"]["n_fft"],
        rnn_layers=s_cfg["model"]["rnn_layers"],
        rnn_units=s_cfg["model"]["rnn_units"],
        kernel_num=s_cfg["model"]["kernel_num"],
        kernel_size=s_cfg["model"]["kernel_size"],
    )

    trainer = KD_M_trainer(s_cfg, teacher_model, model, train_iter, valid_iter, device)
    trainer()
    pass