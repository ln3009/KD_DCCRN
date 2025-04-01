# -*- coding: utf-8 -*-

import os
import zipfile
import torch
import torch.nn as nn
from tqdm import tqdm


def unzip(zip_path, unzip_path=None):
    """解压 ZIP 文件

    Args:
        zip_path (str): ZIP 文件路径
        unzip_path (str, optional): 解压目标路径（默认与 ZIP 文件同名的文件夹）

    Returns:
        unzip_path: 解压目标路径
    """
    # check zip file
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"文件 {zip_path} 不存在！")

    # set unzip path
    if unzip_path == None:
        unzip_path = os.path.splitext(zip_path)[0]

    # unzip
    with zipfile.ZipFile(zip_path) as zf:
        for file in tqdm(zf.infolist(), desc="unzip..."):
            try:
                zf.extract(file, unzip_path)
            except zipfile.error as e:
                print(e)

    return unzip_path


def prepare_empty_path(paths, resume=False):
    """prepare empty path

    Args:
        paths (list): 目录路径列表
        resume (bool, optional): 是否为恢复训练模式，若为True则必须存在目录
    """
    for path in paths:
        if resume:
            # assert os.path.exists(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"路径 {path} 不存在，无法恢复训练！")
        else:
            os.makedirs(path, exist_ok=True) # 如果不存在，则创建目录


def print_size_of_model(model):
    """ 打印 PyTorch 模型大小（MB）

    Args:
        model (torch.nn.Module): 传入的 PyTorch 模型
    """
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def print_networks(models: list):
     """打印多个 PyTorch 模型的参数量

    Args:
        models (list): PyTorch 模型列表
    """
    print(f"Contains {len(models)} models, the number of the parameters is: ")

    params_of_all_networks = 0
    for idx, model in enumerate(models, start=1):
        params_of_network = 0
        for param in model.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


def flatten_parameters(m):
    if isinstance(m, nn.LSTM):
        m.flatten_parameters()
