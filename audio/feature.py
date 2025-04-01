# -*- coding: utf-8 -*-

import numpy as np

EPS = np.finfo(np.float32).eps


#  判断信号是否发生裁剪
def is_clipped(data, clipping_threshold=0.99):
    return any(abs(data) > clipping_threshold)

#  随机截取固定长度的音频片段
def sub_sample(noisy, clean, samples):
    """随机选择固定长度的数据片段

    Args:
        noisy (float): 含噪音的音频数据
        clean (float): 纯净的音频数据
        samples (int): 需要的固定长度

    Returns:
        noisy, clean: 处理后的等长噪声和纯净数据
    """
    length = len(noisy)

    #长音频随机切片，短音频拼补0
    if length > samples:
        start_idx = np.random.randint(length - samples)
        end_idx = start_idx + samples
        noisy = noisy[start_idx:end_idx]
        clean = clean[start_idx:end_idx]
    elif length < samples:
        noisy = np.append(noisy, np.zeros(samples - length))
        clean = np.append(clean, np.zeros(samples - length))
    else:
        pass

    assert len(noisy) == len(clean) == samples

    return noisy, clean
