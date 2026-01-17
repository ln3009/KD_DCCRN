# -*- coding: utf-8 -*-
"""
KD_R_T_trainer.py
按帧在频带上做 softmax 的温度蒸馏实现（频带分布蒸馏，KL × T^2）

假设：
- Teacher/Student 前向都会返回包含 cRM 的输出，形状 [B, 2, F, T]（通道 2=实+虚）。
- 任务损失（task_criterion）由外部传入（如多分辨率 STFT / 复数谱 L1 / SI-SDR 的组合）。
- DataLoader 提供 (noisy, clean) 或字典，外部的 task_criterion 自行解析需要的张量。

你可能需要根据自己工程里的 forward 返回结构，适配 _ExtractMask() 的取数逻辑。
"""
# KD_R_Temperature

import sys
import os
import argparse
import toml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from trainer.base_trainer import BaseTrainer
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from audio.utils import prepare_empty_path

class KD_R_T_KL_Trainer(BaseTrainer):
    """
    温度蒸馏（按帧在频带维做 softmax）版本

    总损失：
      Loss = (1 - v) * student_supervised_loss
             + v * T^2 * KL( log_softmax(student_logit/T), softmax(teacher_logit/T) )

    其中：
      - student_logit / teacher_logit = 对数幅度谱（由 cRM -> 幅度 -> log 得到）
      - KL 在每一帧上沿频带维（F 维）计算；实现上把 [B, F, T] 转为 [B*T, F]
      - T^2 为温度缩放后的常规校正项
    """
    def __init__(self, config, teacher_model, model, train_iter, valid_iter, device="cpu"):
        #  the config is config of student
        super().__init__(config, model, train_iter, valid_iter, device=device)

        # get teacher model
        self.teacher_model = teacher_model         

        # get knowledge_distillation args
        self.teacher_model_path = config["knowledge_distillation"]["teacher_model"]
        self.v = float(config["knowledge_distillation"]["v"])                 # KD weight
        self.T = float(config["knowledge_distillation"]["temperature"])       # KD Temperature

        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "KD_R_T5_KL_x1_16_2")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "KD_R_T5_KL_x1_16_2")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # 加载教师模型
        self.teacher_model = self.teacher_model.to(self.device)
        self.load_teacher_model()
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        # calculate MSE 
        self.mse_loss = nn.MSELoss()

        # 数值稳定常量
        self._eps = 1e-8

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path), f"Teacher model not found: {self.teacher_model_path}"
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

        # 可能存在 "module." 前缀的情况，可按需处理
        self.teacher_model.load_state_dict(state_dict, strict=True)
        print(f"Loaded teacher weights from {self.teacher_model_path} done")        

    @staticmethod
    def _mask_to_log_mag(mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        cRM -> 幅度 -> log 幅度
        输入: mask [B, 2, F, T]  (2=实/虚)
        输出: log_mag [B, F, T]
        """
        real = mask[:, 0]  # [B, F, T]
        imag = mask[:, 1]
        mag = torch.sqrt(real * real + imag * imag + eps)
        log_mag = torch.log(mag + eps)
        return log_mag

    @staticmethod
    def _freq_softmax_kldiv(student_logit: torch.Tensor,
                            teacher_logit: torch.Tensor,
                            temperature: float) -> torch.Tensor:
        """
        在频带维（F 维）做 softmax 的 KL × T^2。
        输入/输出形状：
            student_logit / teacher_logit: [B, F, T] 的“对数幅度 logit”
        计算：
            先转为 [B, T, F]，再展平为 [B*T, F]，在 F 维做 softmax/log_softmax。
        """
        # [B, F, T] -> [B, T, F]
        s = student_logit.transpose(1, 2).contiguous()
        t = teacher_logit.transpose(1, 2).contiguous()

        B, T, F = s.shape
        s = s.view(-1, F)  # [B*T, F]
        t = t.view(-1, F)

        temp = float(temperature)
        log_p_s = nn.functional.log_softmax(s / temp, dim=1)  # student: log prob
        p_t = nn.functional.softmax(t / temp, dim=1)          # teacher: prob

        # KLDivLoss expects log-prob as input, prob as target
        kd_kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (temp * temp)
        return kd_kl
    
    @classmethod
    def response_distillation_loss(cls, student_mask: torch.Tensor,
                                   teacher_mask: torch.Tensor,
                                   T: float,
                                   eps: float = 1e-8) -> torch.Tensor:
        """
        频带 softmax 的温度蒸馏（KL × T^2）：
          1) cRM -> 幅度 -> log 幅度，得到对数幅度“logit” [B, F, T]
          2) 在频带维（F 维）逐帧 softmax（实现上把 [B, T, F] 展平到 [B*T, F]）
          3) KD = T^2 * KL( log_softmax(student/T), softmax(teacher/T) )
        """
        s_log = cls._mask_to_log_mag(student_mask, eps=eps)   # [B, F, T]
        t_log = cls._mask_to_log_mag(teacher_mask, eps=eps)   # [B, F, T]
        kd = cls._freq_softmax_kldiv(s_log, t_log, temperature=T)
        return kd

    def train_epoch(self, epoch):
        """
        每个 epoch 里：
          1. 由教师模型（no_grad）生成 teacher_mask（[B, 2, F, T]）
          2. 由学生模型生成 student_mask
          3. student_supervised_loss = -SI_SDR(enh_student, clean) 或 你的 task loss
          4. distill_loss = T^2 * KL( log_softmax(student_logit/T), softmax(teacher_logit/T) )
          5. 总损失 = (1 - v) * supervised_loss + v * distill_loss
        """
        loss_total = 0.0
        student_loss_total = 0.0

        for noisy, clean in tqdm(self.train_iter, desc="Train KD_R_T"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            with torch.no_grad():
                teacher_mask = self.teacher_model(noisy_spec)   # [B, 2, F, T]
                teacher_mask = teacher_mask.detach()

            with autocast(enabled=self.use_amp):
                student_mask = self.model(noisy_spec)           # [B, 2, F, T]
            
            student_enh = self.audio_istft(student_mask, noisy_spec)
            student_loss = self.loss(student_enh, clean)     

            # 温度蒸馏（频带 softmax，逐帧 KL × T^2）
            distill_loss = self.response_distillation_loss(student_mask, teacher_mask, self.T, eps=self._eps)

            loss = (1.0 - self.v) * student_loss + self.v * distill_loss
            
            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            student_loss_total += float(student_loss.item())
            loss_total += float(loss.item())

        avg_student = student_loss_total / len(self.train_iter)
        avg_loss = loss_total / len(self.train_iter)
        print(f"[Epoch {epoch}]  Train Loss: {avg_loss:.4f}  |  Student loss: {avg_student:.4f}")
        self.update_scheduler(avg_loss)

        self.writer.add_scalar("loss/student", avg_student, epoch)
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)     

if __name__ == "__main__":
    '''
    python trainer/KD_R_T_KL_trainer_copy.py   -TC config/base_config.toml   -SC config/KD_R_T_config.toml  --gpu 2
    '''
    parser = argparse.ArgumentParser(description="Knowledge Distillation with Response & Temperature (Freq-Softmax KL)")
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
    config = toml.load(args.teacher_config)

    # 构建教师模型结构（与 BaseTrainer 相同）
    teacher_model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
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
    trainer = KD_R_T_KL_Trainer(config, teacher_model, model, train_iter, valid_iter, device)
    trainer()

    pass