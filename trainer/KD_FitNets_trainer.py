# -*- coding: utf-8 -*-
# KD_FitNets

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

    其中      
      feature_distillation_loss = mean_i MSE(s_feat_i, t_feat_i)
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
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "fitnets_x1_4_1_2")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "fitnets_x1_4_1_2")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)
        
        # to device
        self.teacher_model = self.teacher_model.to(self.device)
        # load teacher model
        self.load_teacher_model()
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        
        # 动态构建 adapters（1×1 conv）以对齐 student/teacher 特征通道
        # 先用 dummy 数据一次前向拿到特征 shapes
        n_fft = config["dataset"]["n_fft"]
        Fbins = n_fft // 2 + 1
        Tbins = 100
        dummy = torch.randn(1, Fbins, Tbins, 2, device=self.device)

        with torch.no_grad():
            _, t_feats = self.teacher_model(dummy, return_features=True)
            _, s_feats = self.model(dummy,   return_features=True)

        # 为每一层特征创建一个 1×1 Conv 适配器
        self.adapters = nn.ModuleList()
        for fs, ft in zip(s_feats, t_feats):
            C_s = fs.shape[1]
            C_t = ft.shape[1]
            self.adapters.append(nn.Conv2d(C_s, C_t, kernel_size=1))
        
        for adapter in self.adapters:
            adapter.to(self.device)

        # 将 adapters 的参数加入 optimizer
        self.optimizer.add_param_group({
            "params": self.adapters.parameters()
        })

        # 新增的 param_group 会让 scheduler.param_groups 比 base_lrs/min_lrs 多 1，
        # 因此这里手动扩展这两份列表，保持长度一致。
        if hasattr(self.scheduler, "base_lrs"):
            # 新 group 的初始 lr
            new_lr = self.optimizer.param_groups[-1]['lr']
            self.scheduler.base_lrs.append(new_lr)
        if hasattr(self.scheduler, "min_lrs"):
            # 复用第一个 group 的 min_lr
            self.scheduler.min_lrs.append(self.scheduler.min_lrs[0])

        print_networks(list(self.adapters))

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path), f"Teacher model not found: {self.teacher_model_path}"
        load_model = torch.load(self.teacher_model_path, map_location="cpu")
        self.teacher_model.load_state_dict(load_model)

        print(f"Loaded teacher weights from {self.teacher_model_path} done")

    def feature_distillation_loss(self,s_feats, t_feats):
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
        adapter_path = os.path.join(
            self.checkpoints_path, prefix +"_adapters.pth"
        )
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

        for noisy, clean in tqdm(self.train_iter, desc="Train FitNets"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)  
            
            with torch.no_grad():
                t_mask, t_feats = self.teacher_model(noisy_spec, return_features=True) # [B, 2, F, T]

            with autocast(enabled=self.use_amp):
                s_mask, s_feats = self.model(noisy_spec, return_features=True) # [B, 2, F, T]

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

            student_loss_total += student_loss.item()
            feat_loss_total += feat_loss.item()
            loss_total += loss.item()
        
        avg_student = student_loss_total / len(self.train_iter)
        avg_feat = feat_loss_total / len(self.train_iter)
        avg_loss = loss_total / len(self.train_iter)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Sup: {avg_student:.4f} | Feat: {avg_feat:.4f}")
        self.update_scheduler(avg_loss)

        self.writer.add_scalar("loss/student", avg_student, epoch)
        self.writer.add_scalar("loss/feature", avg_feat, epoch)
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)     



if __name__ == "__main__":
    '''
    python trainer/KD_FitNets_trainer.py   -TC config/base_config.toml   -SC config/KD_FitNets_config.toml  --gpu 0

    tensorboard --logdir logs/train/fitnets_x1_4
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
    config = toml.load(args.teacher_config)
    teacher_model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # get config
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