# eval_metrics.py

import sys
import os
import argparse
import random
import toml
import numpy as np
# 为 librosa 兼容 numpy>=1.24 做补丁
if not hasattr(np, "complex"):
    np.complex = complex
import pandas as pd
from tqdm import tqdm
import gc # 手动删除

import torch
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader

import librosa
from dataset.compute_metrics import compute_metric
from module.dc_crn import DCCRN
from dataset.dataset import DNS_Dataset

EPS = np.finfo(np.float32).eps

def audio_stft(audio, n_fft, hop_len, win_len, window):
    """Compute STFT, return real+imag channels."""
    return torch.stft(
        audio, n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        return_complex=False,
    )

def audio_istft(mask, spec, n_fft, hop_len, win_len, window):
    """Inverse STFT using mask and original spectrogram."""
    mask_mag = (mask[:,0]**2 + mask[:,1]**2).sqrt()
    pr = mask[:,0] / (mask_mag + EPS)
    pi = mask[:,1] / (mask_mag + EPS)
    mask_phase = torch.atan2(pi, pr)
    mask_mag = torch.tanh(mask_mag)

    spec_r, spec_i = spec.unbind(-1)
    spec_mag = (spec_r**2 + spec_i**2).sqrt()
    spec_phase = torch.atan2(spec_i, spec_r)

    enh_mag = mask_mag * spec_mag
    enh_phase = spec_phase + mask_phase

    real = enh_mag * torch.cos(enh_phase)
    imag = enh_mag * torch.sin(enh_phase)
    cspec = torch.complex(real, imag)

    audio = torch.istft(
        cspec, n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        return_complex=False,
    )
    return torch.clamp(audio, -1.0, 1.0)

def evaluate_model(config_path,gpu_id):
    # config
    runs = 2
    save_path = "./mytest/eval_metrics_KD_F_fitnets_x1_16_0.9_2.csv"
    model_path = "../model/dccrn/checkpoints/fitnets_x1_16_0.9_2/best_model.pth"

    # 1) Load config
    cfg = toml.load(config_path)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    # 2) Build model & load weights
    model = DCCRN(
        n_fft=cfg["dataset"]["n_fft"],
        rnn_layers=cfg["model"]["rnn_layers"],
        rnn_units=cfg["model"]["rnn_units"],
        kernel_num=cfg["model"]["kernel_num"],
        kernel_size=cfg["model"]["kernel_size"],
    )
    
    chk = torch.load(model_path, map_location="cpu")    
    model.load_state_dict(chk)
    model = model.to(device).eval()

    # 3) STFT params
    n_fft   = cfg["dataset"]["n_fft"]
    win_len = cfg["dataset"]["win_len"]
    hop_len = cfg["dataset"]["hop_len"]
    window  = torch.hann_window(win_len, periodic=False, device=device)
    sr      = cfg["dataset"]["sr"]

    # 4) DataLoader params
    bs_train = cfg["dataloader"]["batch_size"][0]
    bs_eval  = cfg["dataloader"]["batch_size"][1]
    num_workers = cfg["dataloader"]["num_workers"]
    drop_last   = cfg["dataloader"]["drop_last"]
    pin_memory  = cfg["dataloader"]["pin_memory"]
    n_folds = cfg.get("ppl", {}).get("n_folds", 1)
    n_jobs  = cfg.get("ppl", {}).get("n_jobs", 8)

    results = {}
    dataset_csv_dir = "./dataset_csv"

    @torch.no_grad()
    def infer_batch(noisy_batch):
        spec = audio_stft(noisy_batch, n_fft, hop_len, win_len, window)
        mask = model(spec)
        enh  = audio_istft(mask, spec, n_fft, hop_len, win_len, window)
        # 立即搬到 CPU 并删除 GPU Tensor
        enh_cpu = enh.cpu()
        del spec, mask, enh
        return enh_cpu

    # 5) Loop over splits
    for mode in ["train","valid", "test"]:
        # a) create dataset & DataLoader
        ds = DNS_Dataset(dataset_csv_dir, cfg, mode)
        bs = bs_train if mode == "train" else bs_eval

        if mode == "train":
            num_runs = runs
        else:
            num_runs = 1

        for run in range(1, num_runs+1):

            torch.cuda.empty_cache()
            key = f"{mode}_run{run}"
            # b) reproducible shuffle via generator
            g = torch.Generator().manual_seed(run)
            loader = DataLoader(
                ds,
                batch_size=bs,
                shuffle=True,
                num_workers=num_workers,
                drop_last=drop_last,
                pin_memory=pin_memory,
                generator=g
            )

            enh_list = []
            clean_list = []

            # c) iterate batches
            for batch in tqdm(loader, desc=f"{mode} run{run}"):
                # 解包并搬到 GPU
                if mode in ["valid", "test"]:
                    noisy, clean, _ = batch
                else:
                    noisy, clean = batch    
                
                noisy = noisy.to(device)
                clean = clean.to(device)    

                # 前向并立刻回收
                enh_cpu = infer_batch(noisy)

                # 收集结果
                for i in range(enh_cpu.size(0)):
                    enh_list.append(enh_cpu[i].numpy())
                    clean_list.append(clean[i].cpu().numpy())

                # 强制回收：把所有中间 Tensor 的引用删掉，触发 GC，再清理 cache
                del noisy, clean, enh_cpu
                gc.collect()
                torch.cuda.empty_cache()

            # d) compute metrics in parallel
            metrics = {m: [] for m in ["SI_SDR", "STOI", "WB_PESQ", "NB_PESQ"]}
            compute_metric(
                enh_list, clean_list, metrics,
                n_folds=n_folds, n_jobs=n_jobs, pre_load=True
            )
            metrics["avg_loss"] = -metrics["SI_SDR"]
            results[key] = metrics

    # 6) Save to CSV
    df_out = pd.DataFrame(results).T
    df_out.to_csv(save_path, float_format="%.6f")
    print("Saved eval_metrics.csv")
    print(df_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DCCRN via DNS_Dataset")
    parser.add_argument("-C", "--config", required=True, help="Path to config TOML")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="指定要使用的 GPU 编号 (例如 0, 1, 2, …)，如果只有 CPU 可用则忽略。"
    )

    ## python mytest/val_metrics.py -C config/base_config_test.toml --gpu 2
    args = parser.parse_args()

    evaluate_model(args.config, args.gpu)