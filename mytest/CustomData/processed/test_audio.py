
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
print("\n播放干净语音:")
ipd.display(ipd.Audio(clean_audio, rate=sr))
print("\n播放带噪语音:")
ipd.display(ipd.Audio(noisy_audio, rate=sr))
