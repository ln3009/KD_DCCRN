import sys
import os
import glob
import argparse
import toml

import numpy as np
import pandas as pd
import librosa
from speechmos import dnsmos
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
sys.path.append(os.getcwd())

from module.dc_crn import DCCRN

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
    # mask [B, 2, F, T]
    # spec [B, F, T, 2]
    mask_mags = (mask[:, 0, :, :] ** 2 + mask[:, 1, :, :] ** 2) ** 0.5
    phase_real = mask[:, 0, :, :] / (mask_mags + EPS)
    phase_imag = mask[:, 1, :, :] / (mask_mags + EPS)
    mask_phase = torch.atan2(phase_imag, phase_real)
    mask_mags = torch.tanh(mask_mags)
    enh_mags = mask_mags * torch.sqrt(spec[:, :, :, 0] ** 2 + spec[:, :, :, 1] ** 2)
    enh_phase = torch.atan2(spec[:, :, :, 1], spec[:, :, :, 0]) + mask_phase
    spec_real = enh_mags * torch.cos(enh_phase)
    spec_imag = enh_mags * torch.sin(enh_phase)
    # [B, F, T]
    cspec = spec_real + 1j * spec_imag
    # [B, S]
    audio = torch.istft(
        cspec, n_fft,
        hop_length=hop_len,
        win_length=win_len,
        window=window,
        return_complex=False,
    )
    audio = torch.clamp(audio, min=-1.0, max=1.0)
    return audio


def infer_enhance(model, audio_tensor, cfg, device):
    # audio_tensor: 1D torch.FloatTensor on device
    n_fft = cfg["stft"]["n_fft"]
    win_len = cfg["stft"]["win_len"]
    hop_len = cfg["stft"]["hop_len"]
    window = torch.hann_window(win_len, periodic=False, device=device)

    spec = audio_stft(audio_tensor, n_fft, hop_len, win_len, window)  # [F, T, 2]
    mask = model(spec.unsqueeze(0))  # mask [B, 2, F, T]  # spec [B, F, T, 2]        
    enh = audio_istft(mask, spec.unsqueeze(0), n_fft, hop_len, win_len, window)
    enh = enh.squeeze(0)   
    return enh


def plot_spectrogram(audio, sr, out_path, n_fft=512, hop_len=128):
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_len))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure()
    librosa.display.specshow(D_db, sr=sr, hop_length=hop_len, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_dnsmos(audio_path, dns_model=None):
    """
    用 speechmos.dnsmos.run 来计算 MOS 分：
      - ovrl_mos: Overall quality
      - sig_mos: Signal quality
      - bak_mos: Background noise quality
    """
    # 1) load audio（librosa 会自动归一化到 [-1,1]）
    audio, sr = librosa.load(audio_path, sr=None)
    # 2) 调用 dnsmos
    res = dnsmos.run(audio, sr=sr)
    # res 示例：
    # {
    #  'filename': '.../enh.wav',
    #  'ovrl_mos': 2.21,
    #  'sig_mos': 3.29,
    #  'bak_mos': 2.14,
    #  'p808_mos': 3.07
    # }
    return {
        'OVRL': res['ovrl_mos'],
        'SIG':  res['sig_mos'],
        'BAK':  res['bak_mos'],
        'P808': res['p808_mos']
    }


def load_dccrn(checkpoint_path, cfg, device):
    model = DCCRN(
        n_fft=cfg["dataset"]["n_fft"],
        rnn_layers=cfg["model"]["rnn_layers"],
        rnn_units=cfg["model"]["rnn_units"],
        kernel_num=cfg["model"]["kernel_num"],
        kernel_size=cfg["model"]["kernel_size"],
    )
    chk = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(chk)
    model.to(device).eval()
    return model
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试各模型 DNSMOS 指标(3quest噪声场景主麦信号)')
    parser.add_argument('-C', '--config', required=True, help='路径到配置 TOML 文件')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 编号')
    args = parser.parse_args()

    ## python DNSMOStest/dnsmos_test.py  -C DNSMOStest/config.toml --gpu 0

    cfg = toml.load(args.config)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    # 路径配置
    data_root = cfg['paths']['data_root']  # ../data/cws
    noisy_folder = os.path.join(data_root, cfg['paths']['noisy_folder'])  # cws_talk
    model_root = cfg['paths']['model_root']  # ../model/dccrn/checkpoints
    output_root = os.path.join(cfg['paths']['output_path'])  # ../data/cws/output

    os.makedirs(output_root, exist_ok=True)

    # 如果需要，可在此加载 DNSMOS 模型
    # dns_model = DNSMOS(cfg['dnsmos']['model_path'], device=device)
    dns_model = None

    results = []
    '''
    # 基线(noisy)评估
    baseline_folder = os.path.join(output_root, 'baseline')
    os.makedirs(baseline_folder, exist_ok=True)
    spec_base = os.path.join(baseline_folder, 'specs')
    os.makedirs(spec_base, exist_ok=True)
    audio_base = os.path.join(baseline_folder, 'audio')
    os.makedirs(audio_base, exist_ok=True)
    
    for wav_path in tqdm(glob.glob(os.path.join(noisy_folder, '*.wav')), desc='Baseline 音频'):
        wav_name = os.path.basename(wav_path)
        y, sr = librosa.load(wav_path, sr=cfg['dataset']['sr'])
        # 保存原始音频
        sf.write(os.path.join(audio_base, wav_name), y, sr)
        # 绘制谱图
        plot_spectrogram(y, sr, os.path.join(spec_base, wav_name.replace('.wav', '.png')),
                         n_fft=cfg['stft']['n_fft'], hop_len=cfg['stft']['hop_len'])
        # 计算 DNSMOS
        scores = compute_dnsmos(os.path.join(audio_base, wav_name), dns_model)
        results.append({'model': 'noisy', 'file': wav_name, **scores})
    '''
    model_list = cfg['paths'].get('model_list', [])
    # 各模型评估
    for model_name in model_list:
        model_dir = os.path.join(model_root, model_name)
        checkpoint = os.path.join(model_dir, 'best_model.pth')
        if not os.path.isfile(checkpoint):
            print(f"[Warning] 找不到 {checkpoint}, 跳过")
            continue
        
        model = load_dccrn(checkpoint, cfg, device)

        # 创建输出子目录
        m_out = os.path.join(output_root, model_name)
        spec_out = os.path.join(m_out, 'specs')
        audio_out = os.path.join(m_out, 'audio')
        os.makedirs(spec_out, exist_ok=True)
        os.makedirs(audio_out, exist_ok=True)

        for wav_path in tqdm(glob.glob(os.path.join(noisy_folder, '*.wav')), desc=f'{model_name} 推理'):
            wav_name = os.path.basename(wav_path)
            y, sr = librosa.load(wav_path, sr=cfg['dataset']['sr'])
            tensor = torch.from_numpy(y).float().to(device)
            with torch.no_grad():
                enh_nograd = infer_enhance(model, tensor, cfg, device)

            enh = enh_nograd.cpu().numpy()

            # 保存增强后音频
            sf.write(os.path.join(audio_out, wav_name), enh, sr)
            # 绘制增强后谱图
            plot_spectrogram(enh, sr, os.path.join(spec_out, wav_name.replace('.wav', '.png')),
                             n_fft=cfg['stft']['n_fft'], hop_len=cfg['stft']['hop_len'])
            # 计算 DNSMOS
            scores = compute_dnsmos(os.path.join(audio_out, wav_name), dns_model)
            results.append({'model': model_name, 'file': wav_name, **scores})

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_root, 'DNSMOS_x1_16.csv'), index=False)
    print('完成，结果已保存到', os.path.join(output_root, 'DNSMOS_x1_16.csv'))
