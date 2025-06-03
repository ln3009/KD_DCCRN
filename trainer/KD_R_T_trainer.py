# -*- coding: utf-8 -*-
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

class KD_R_T_Trainer(BaseTrainer):
    """    
    Loss = (1 - v) * student_supervised_loss + v * T^2 * MSE(student_mask, teacher_mask)
    """
    def __init__(self, config, teacher_model, model, train_iter, valid_iter, device="cpu"):
        #  the config is config of student
        super().__init__(config, model, train_iter, valid_iter, device=device)

        # get teacher model
        self.teacher_model = teacher_model         

        # get knowledge_distillation args     
        self.teacher_model_path = config["knowledge_distillation"]["teacher_model"]
        self.v = config["knowledge_distillation"]["v"]     # KD weight
        self.T = config["knowledge_distillation"]["temperature"]    # KD Temperature

        # reconfig path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "KD_R_T_x1_4")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "KD_R_T_x1_4")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # 加载教师模型
         # to device
        self.teacher_model = self.teacher_model.to(self.device)
        # load teacher model
        self.load_teacher_model()

        self.teacher_model.eval()

        # calculate MSE 
        self.mse_loss = nn.MSELoss() 

    def load_teacher_model(self):
        assert os.path.exists(self.teacher_model_path), f"Teacher model not found: {self.teacher_model_path}"
        load_model = torch.load(self.teacher_model_path, map_location="cpu")
        self.teacher_model.load_state_dict(load_model)

        print(f"Loaded teacher weights from {self.teacher_model_path} done")

    @staticmethod
    def response_distillation_loss(student_mask, teacher_mask, T):
        """        
        distill_loss = T^2 * MSE(student_mask, teacher_mask)
        省略对 logits 做 Softmax —— 因为 mask 本身是实值连续
        """
        mse = nn.functional.mse_loss(student_mask, teacher_mask)
        return (T ** 2) * mse

    def train_epoch(self, epoch):
        """
        每个 epoch 里：
          1. 用教师模型生成 teacher_mask（no_grad）
          2. 用学生模型生成 student_mask
          3. student_supervised_loss = -SI_SDR(enh_student, clean)
          4. distill_loss = T^2 * MSE(student_mask, teacher_mask)
          5. 总损失 = (1 - alpha) * supervised_loss + alpha * distill_loss
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

            with autocast(enabled=self.use_amp):
                student_mask = self.model(noisy_spec)          # [B, 2, F, T]
            
            student_enh  = self.audio_istft(student_mask, noisy_spec)           
            student_loss = self.loss(student_enh, clean)            
            distill_loss = self.response_distillation_loss(student_mask, teacher_mask, self.T)
            loss = (1 - self.v) * student_loss + self.v * distill_loss
            
            # backward 
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            student_loss_total += student_loss.item() 
            loss_total += loss.item()

        avg_student = student_loss_total / len(self.train_iter)
        avg_loss = loss_total / len(self.train_iter)   
        print(f"[Epoch {epoch}]  Train Loss: {avg_loss:.4f}  |  Student loss: {avg_student:.4f}")
        self.update_scheduler(avg_loss)

        self.writer.add_scalar("loss/student", avg_student, epoch)
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)     

if __name__ == "__main__":
    '''
    python trainer/KD_R_T_trainer.py   -TC config/base_config.toml   -SC config/KD_R_T_config.toml  --gpu 1
    '''
    parser = argparse.ArgumentParser(description="Knowledge Distillation with Response & Temperature")
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
    trainer = KD_R_T_Trainer(config, teacher_model, model, train_iter, valid_iter, device)
    trainer()

    pass