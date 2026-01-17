import librosa
import soundfile as sf
import numpy as np
import os
import random
from pathlib import Path


def process_audio_files():
    """
    处理音频文件：提取干净语音片段，与噪声混合生成带噪语音
    """
    # 设置参数
    start_time = 10  # 起始时间（秒）
    duration = 20    # 持续时间（秒）
    target_sr = 16000  # 目标采样率
    snr_db = 5       # 信噪比（dB），可调整
    
    # 文件路径
    base_dir = Path("CustomData")
    clean_audio_path = base_dir / "Fight of Your Life.mp3"
    noise_audio_path = base_dir / "babble.wav"
    
    # 创建输出文件夹
    output_dir = base_dir / "processed"
    output_dir.mkdir(exist_ok=True)
    
    print("开始处理音频文件...")
    
    # 1. 处理干净语音
    print("1. 处理干净语音...")
    try:
        # 加载MP3文件，从10秒处开始，读取20秒
        clean_audio, sr_clean = librosa.load(
            clean_audio_path,
            sr=target_sr,  # 直接以目标采样率加载
            offset=start_time,
            duration=duration
        )
        
        # 保存为WAV格式
        clean_output_path = output_dir / "clean_speech.wav"
        sf.write(clean_output_path, clean_audio, target_sr)
        print(f"   干净语音已保存: {clean_output_path}")
        print(f"   采样率: {target_sr}Hz, 时长: {len(clean_audio)/target_sr:.2f}秒")
        
    except Exception as e:
        print(f"   处理干净语音时出错: {e}")
        return
    
    # 2. 处理噪声
    print("2. 处理噪声...")
    try:
        # 加载噪声文件
        noise_audio, sr_noise = librosa.load(noise_audio_path, sr=None)
        
        # 如果噪声采样率不是16kHz，则重采样
        if sr_noise != target_sr:
            noise_audio = librosa.resample(noise_audio, orig_sr=sr_noise, target_sr=target_sr)
            sr_noise = target_sr
        
        # 随机选择20秒噪声片段
        noise_length = len(noise_audio)
        segment_samples = duration * target_sr
        
        if noise_length >= segment_samples:
            # 如果噪声长度足够，随机选择起始点
            max_start = noise_length - segment_samples
            start_idx = random.randint(0, max_start)
            noise_segment = noise_audio[start_idx:start_idx + segment_samples]
        else:
            # 如果噪声长度不足，循环填充
            repeats = int(np.ceil(segment_samples / noise_length))
            repeated_noise = np.tile(noise_audio, repeats)
            noise_segment = repeated_noise[:segment_samples]
            print(f"   注意: 噪声文件长度不足，已循环填充")
        
        # 保存噪声片段
        noise_output_path = output_dir / "noise_segment.wav"
        sf.write(noise_output_path, noise_segment, target_sr)
        print(f"   噪声片段已保存: {noise_output_path}")
        
    except Exception as e:
        print(f"   处理噪声时出错: {e}")
        return
    
    # 3. 合成带噪语音
    print("3. 合成带噪语音...")
    try:
        # 确保两个音频长度相同
        min_length = min(len(clean_audio), len(noise_segment))
        clean_audio = clean_audio[:min_length]
        noise_segment = noise_segment[:min_length]
        
        # 计算功率
        clean_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_segment ** 2)
        
        if noise_power > 0:
            # 根据SNR计算需要的噪声缩放因子
            target_snr_linear = 10 ** (snr_db / 10)
            target_noise_power = clean_power / target_snr_linear
            
            # 缩放噪声
            scale_factor = np.sqrt(target_noise_power / noise_power)
            scaled_noise = noise_segment * scale_factor
            
            # 混合音频
            noisy_audio = clean_audio + scaled_noise
            
            # 归一化，防止裁剪
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val * 0.99
            
            # 计算实际SNR
            actual_noise_power = np.mean(scaled_noise ** 2)
            actual_snr = 10 * np.log10(clean_power / actual_noise_power)
            print(f"   目标SNR: {snr_db:.1f}dB, 实际SNR: {actual_snr:.1f}dB")
            
        else:
            print("   警告: 噪声功率为0，无法合成带噪语音")
            return
        
        # 保存带噪语音
        noisy_output_path = output_dir / f"noisy_speech_snr{snr_db}dB.wav"
        sf.write(noisy_output_path, noisy_audio, target_sr)
        print(f"   带噪语音已保存: {noisy_output_path}")
        
    except Exception as e:
        print(f"   合成带噪语音时出错: {e}")
        return
    
    # 4. 生成信息文件
    info_path = output_dir / "processing_info.txt"
    with open(info_path, 'w') as f:
        f.write("=== 音频处理信息 ===\n")
        f.write(f"干净语音文件: {clean_audio_path.name}\n")
        f.write(f"噪声文件: {noise_audio_path.name}\n")
        f.write(f"采样率: {target_sr} Hz\n")
        f.write(f"片段时长: {duration} 秒\n")
        f.write(f"干净语音起始时间: {start_time} 秒\n")
        f.write(f"目标信噪比: {snr_db} dB\n")
        f.write(f"实际信噪比: {actual_snr:.2f} dB\n")
        f.write(f"生成文件:\n")
        f.write(f"  1. {clean_output_path.name} (干净语音)\n")
        f.write(f"  2. {noise_output_path.name} (噪声片段)\n")
        f.write(f"  3. {noisy_output_path.name} (带噪语音)\n")
    
    print(f"\n处理完成！所有文件已保存到: {output_dir}")
    print(f"详细信息已保存到: {info_path}")
    
    return {
        'clean_audio': clean_audio,
        'noise_segment': noise_segment,
        'noisy_audio': noisy_audio,
        'clean_path': clean_output_path,
        'noise_path': noise_output_path,
        'noisy_path': noisy_output_path,
        'sr': target_sr,
        'snr': actual_snr
    }

def generate_test_script():
    """生成测试脚本，用于验证生成的音频文件"""
    test_script = '''
# 音频验证脚本
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载生成的音频文件
base_dir = Path("CustomData/processed")
clean_audio, sr = sf.read(base_dir / "clean_speech.wav")
noisy_audio, _ = sf.read(base_dir / "noisy_speech_snr5dB.wav")

# 绘制波形图
plt.figure(figsize=(12, 8))

# 干净语音波形
plt.subplot(3, 1, 1)
time = np.arange(len(clean_audio)) / sr
plt.plot(time, clean_audio, 'b', alpha=0.7)
plt.title("干净语音波形")
plt.xlabel("时间 (秒)")
plt.ylabel("幅度")
plt.grid(True, alpha=0.3)

# 带噪语音波形
plt.subplot(3, 1, 2)
time = np.arange(len(noisy_audio)) / sr
plt.plot(time, noisy_audio, 'r', alpha=0.7)
plt.title("带噪语音波形 (SNR=5dB)")
plt.xlabel("时间 (秒)")
plt.ylabel("幅度")
plt.grid(True, alpha=0.3)

# 绘制语谱图
plt.subplot(3, 1, 3)
D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("带噪语音语谱图")
plt.tight_layout()
plt.savefig("CustomData/processed/audio_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("音频验证完成！已保存对比图到 CustomData/processed/audio_comparison.png")

# 播放音频（如果环境支持）
import IPython.display as ipd
print("\\n播放干净语音:")
ipd.display(ipd.Audio(clean_audio, rate=sr))
print("\\n播放带噪语音:")
ipd.display(ipd.Audio(noisy_audio, rate=sr))
'''
    
    test_script_path = Path("CustomData/processed/test_audio.py")
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    
    print(f"测试脚本已生成: {test_script_path}")
    print("运行 'python CustomData/processed/test_audio.py' 来验证结果")

if __name__ == "__main__":
    # 确保所需库已安装
    try:
        import librosa
        import soundfile
    except ImportError:
        print("缺少必要的库，正在安装...")
        import subprocess
        subprocess.check_call(["pip", "install", "librosa", "soundfile", "numpy"])
        import librosa
        import soundfile as sf
    
    # 运行主处理函数
    result = process_audio_files()
    
    if result:
        # 生成测试脚本
        generate_test_script()
        
        print("\n" + "="*50)
        print("下一步：")
        print("1. 运行生成的测试脚本验证结果: python CustomData/processed/test_audio.py")
        print("2. 您可以在 CustomData/processed/ 文件夹中找到所有生成的文件")
        print("3. 如需调整参数（如SNR），请修改代码中的 snr_db 变量")
        print("="*50)