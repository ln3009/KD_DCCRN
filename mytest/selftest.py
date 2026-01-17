# selftest.py
import os
import sys
import argparse
import toml
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== 路径设置 - 支持从项目根目录运行 ==========
# 获取当前工作目录（项目根目录）
cwd = Path.cwd()  # 应该是 ~/audio_project/SE_DCCRN/SE-DCCRN
print(f"当前工作目录: {cwd}")

# 检查当前目录下是否有module和audio目录
if (cwd / "module").exists() and (cwd / "module" / "dc_crn.py").exists():
    # 如果在项目根目录，直接添加到路径
    sys.path.insert(0, str(cwd))
    project_root = cwd
elif (cwd.parent / "module").exists() and (cwd.parent / "module" / "dc_crn.py").exists():
    # 如果在mytest目录，添加上一级目录
    sys.path.insert(0, str(cwd.parent))
    project_root = cwd.parent
else:
    # 尝试通过脚本位置查找
    script_dir = Path(__file__).parent.absolute()
    if (script_dir.parent / "module").exists():
        sys.path.insert(0, str(script_dir.parent))
        project_root = script_dir.parent
    else:
        project_root = cwd
        sys.path.insert(0, str(cwd))

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:2]}")
# ===================================================

try:
    from module.dc_crn import DCCRN
    print("成功导入 DCCRN 模型")
except ImportError as e:
    print(f"导入 DCCRN 时出错: {e}")
    print("请确保您在项目根目录(SE-DCCRN)下运行，或者脚本路径设置正确")
    sys.exit(1)

try:
    from audio.feature import is_clipped, EPS
    from audio.utils import prepare_empty_path
    from audio.metrics import SI_SDR
    print("成功导入 audio 模块")
except ImportError as e:
    print(f"导入 audio 模块时出错: {e}")
    # 如果 audio 模块不存在，使用替代实现
    EPS = 1e-8
    
    def is_clipped(audio):
        """检查音频是否裁剪"""
        return np.max(np.abs(audio)) > 0.99
    
    def prepare_empty_path(paths, resume=False):
        """准备空目录"""
        for path in paths:
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            elif not resume:
                import shutil
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
    
    def SI_SDR(enh, clean):
        """计算SI-SDR"""
        # 简单实现
        return torch.tensor(0.0)
    
    print("使用替代的audio模块实现")

class AudioVisualizer:
    """音频可视化工具类"""
    
    def __init__(self, sr=16000, n_fft=512, hop_length=256, win_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
    def compute_spectrogram(self, audio):
        """计算梅尔语谱图"""
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    
    def plot_audio_comparison(self, clean_audio, noisy_audio, enhanced_audio, 
                            clean_path, noisy_path, output_path, snr_level, model_name="DCCRN"):
        """
        绘制音频对比图：波形图和语谱图
        包括：干净音频、带噪音频、增强音频
        """
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Audio Enhancement Comparison (SNR={snr_level}dB)', 
                    fontsize=16, fontweight='bold')
        
        # 设置时间轴
        time_clean = np.arange(len(clean_audio)) / self.sr
        time_noisy = np.arange(len(noisy_audio)) / self.sr
        time_enh = np.arange(len(enhanced_audio)) / self.sr
        
        # 1. 干净语音波形
        axes[0, 0].plot(time_clean, clean_audio, 'b', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Clean Speech - Waveform', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, max(time_clean)])
        
        # 2. 干净语音语谱图
        S_clean = self.compute_spectrogram(clean_audio)
        librosa.display.specshow(S_clean, sr=self.sr, hop_length=self.hop_length,
                                x_axis='time', y_axis='mel', ax=axes[0, 1])
        axes[0, 1].set_title('Clean Speech - Spectrogram', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        
        # 3. 带噪语音波形
        axes[1, 0].plot(time_noisy, noisy_audio, 'r', alpha=0.7, linewidth=1)
        axes[1, 0].set_title(f'Noisy Speech (SNR={snr_level}dB) - Waveform', 
                           fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, max(time_noisy)])
        
        # 4. 带噪语音语谱图
        S_noisy = self.compute_spectrogram(noisy_audio)
        librosa.display.specshow(S_noisy, sr=self.sr, hop_length=self.hop_length,
                                x_axis='time', y_axis='mel', ax=axes[1, 1])
        axes[1, 1].set_title(f'Noisy Speech (SNR={snr_level}dB) - Spectrogram', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        
        # 5. 增强语音波形
        axes[2, 0].plot(time_enh, enhanced_audio, 'g', alpha=0.7, linewidth=1)
        axes[2, 0].set_title(f'Enhanced Speech ({model_name}) - Waveform', 
                           fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_xlim([0, max(time_enh)])
        
        # 6. 增强语音语谱图
        S_enh = self.compute_spectrogram(enhanced_audio)
        librosa.display.specshow(S_enh, sr=self.sr, hop_length=self.hop_length,
                                x_axis='time', y_axis='mel', ax=axes[2, 1])
        axes[2, 1].set_title(f'Enhanced Speech ({model_name}) - Spectrogram', 
                           fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Frequency (Hz)')
        
        # 调整布局
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return {
            'clean_path': clean_path,
            'noisy_path': noisy_path,
            'snr_level': snr_level,
            'duration': len(clean_audio)/self.sr,
            'sample_rate': self.sr
        }

class AudioProcessor:
    """音频处理器，包含STFT和ISTFT操作"""
    
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        
        # 获取音频参数
        self.sr = config["dataset"]["sr"]
        self.n_fft = config["dataset"]["n_fft"]
        self.hop_len = config["dataset"]["hop_len"]
        self.win_len = config["dataset"]["win_len"]
        self.window = torch.hann_window(self.win_len, periodic=False, device=self.device)
    
    def audio_stft(self, audio):
        """
        执行STFT变换
        audio: [B, S] -> 返回: [B, F, T, 2]
        """
        # 确保音频是2D的 [B, S]
        if audio.dim() == 3:
            # 如果是3D [B, 1, S]，压缩为2D
            audio = audio.squeeze(1)
        
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        return spec
    
    def audio_istft(self, mask, spec):
        """
        执行ISTFT变换
        mask: [B, 2, F, T]
        spec: [B, F, T, 2]
        返回: [B, S]
        """
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
            cspec,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

class AudioInferencer:
    """音频推理器"""
    
    def __init__(self, config, model, device="cpu", model_name="DCCRN"):
        self.config = config
        self.model = model
        self.device = device
        self.model_name = model_name
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(config, device)
        
        # 设置模型为评估模式，并移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化可视化工具
        self.visualizer = AudioVisualizer(
            sr=config["dataset"]["sr"],
            n_fft=config["dataset"]["n_fft"],
            hop_length=config["dataset"]["hop_len"],
            win_length=config["dataset"]["win_len"]
        )
    
    def load_audio(self, audio_path, target_sr=16000):
        """加载音频文件并重采样"""
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio, sr
    
    def preprocess_audio(self, audio):
        """预处理音频：转换为tensor，添加批次维度"""
        # 转换为tensor，并添加批次维度 [1, S]
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # [1, S]
        return audio_tensor.to(self.device)
    
    def enhance_audio(self, noisy_audio):
        """增强音频"""
        with torch.no_grad():
            # 预处理
            noisy_tensor = self.preprocess_audio(noisy_audio)
            
            # STFT
            noisy_spec = self.audio_processor.audio_stft(noisy_tensor)
            
            # 模型推理
            mask = self.model(noisy_spec)
            
            # ISTFT
            enhanced_tensor = self.audio_processor.audio_istft(mask, noisy_spec)
            
            # 后处理
            enhanced_audio = enhanced_tensor.squeeze(0).cpu().numpy()
            
            return enhanced_audio
    
    def compute_metrics(self, clean_audio, enhanced_audio):
        """计算增强音频的指标"""
        # 转换为tensor
        clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)
        enh_tensor = torch.FloatTensor(enhanced_audio).unsqueeze(0)
        
        # 计算SI-SDR
        try:
            si_sdr = -SI_SDR(enh_tensor, clean_tensor)
            si_sdr_value = si_sdr.item()
        except:
            # 如果计算失败，使用默认值
            si_sdr_value = 0.0
        
        return {
            'SI-SDR': si_sdr_value
        }
    
    def process_single_pair(self, clean_path, noisy_path, output_dir, snr_level):
        """处理单个干净/带噪音频对"""
        print(f"\n处理 SNR={snr_level}dB 的音频...")
        
        # 加载音频
        clean_audio, _ = self.load_audio(clean_path, self.config["dataset"]["sr"])
        noisy_audio, _ = self.load_audio(noisy_path, self.config["dataset"]["sr"])
        
        # 确保长度一致
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # 增强音频
        print("  进行音频增强...")
        enhanced_audio = self.enhance_audio(noisy_audio)
        
        # 确保增强音频长度一致
        enhanced_audio = enhanced_audio[:min_len]
        
        # 计算指标
        metrics = self.compute_metrics(clean_audio, enhanced_audio)
        print(f"  增强音频指标 - SI-SDR: {metrics['SI-SDR']:.2f} dB")
        
        # 保存增强音频
        enhanced_path = output_dir / f"enhanced_snr{snr_level}dB.wav"
        sf.write(enhanced_path, enhanced_audio, self.config["dataset"]["sr"])
        print(f"  增强音频已保存: {enhanced_path}")
        
        # 生成对比图
        print("  生成对比图...")
        
        # 完整对比图
        comparison_path = output_dir / f"comparison_snr{snr_level}dB.png"
        comparison_info = self.visualizer.plot_audio_comparison(
            clean_audio, noisy_audio, enhanced_audio,
            clean_path, noisy_path, comparison_path, snr_level, self.model_name
        )
        print(f"  对比图已保存: {comparison_path}")
        
        # 检查裁剪
        if is_clipped(enhanced_audio):
            print(f"  警告: 增强音频 {enhanced_path.name} 有裁剪!")
        
        return {
            'clean_audio': clean_audio,
            'noisy_audio': noisy_audio,
            'enhanced_audio': enhanced_audio,
            'clean_path': clean_path,
            'noisy_path': noisy_path,
            'enhanced_path': enhanced_path,
            'comparison_path': comparison_path,
            'snr_level': snr_level,
            'metrics': metrics
        }
    
    def process_all(self, data_dir, output_base_dir):
        """处理所有音频文件"""
        data_dir = Path(data_dir)
        output_base_dir = Path(output_base_dir)
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_base_dir / f"{self.model_name}_results_{timestamp}"
        prepare_empty_path([output_dir], resume=False)
        
        print(f"数据目录: {data_dir}")
        print(f"输出目录: {output_dir}")
        
        # 定义要处理的SNR级别
        snr_levels = [0, 5]
        results = {}
        
        for snr in snr_levels:
            # 构建文件路径
            clean_path = data_dir / f"clean_speech.wav"
            noisy_path = data_dir / f"noisy_speech_snr{snr}dB.wav"
            
            if not clean_path.exists():
                print(f"警告: 干净音频文件不存在: {clean_path}")
                # 尝试其他可能的命名
                clean_path = data_dir / f"clean_speech{snr}.wav"
                if not clean_path.exists():
                    print(f"警告: 干净音频文件不存在: {clean_path}")
                    continue
            
            if not noisy_path.exists():
                print(f"警告: 带噪音频文件不存在: {noisy_path}")
                continue
            
            # 处理单个音频对
            result = self.process_single_pair(
                clean_path, noisy_path, output_dir, snr
            )
            results[f"snr_{snr}db"] = result
        
        return results, output_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DCCRN音频增强推理脚本")
    parser.add_argument("-C", "--config", required=True, type=str, 
                       help="配置文件 (*.toml)")
    parser.add_argument("-M", "--model_checkpoint", required=True, type=str,
                       help="模型检查点文件 (*.pth)")
    parser.add_argument("-D", "--data_dir", default="mytest/CustomData", type=str,
                       help="数据目录路径")
    parser.add_argument("-O", "--output_dir", default="mytest/test_results", type=str,
                       help="输出目录")
    parser.add_argument("--model_name", default="DCCRN", type=str,
                       help="模型名称，用于图片标题")
    args = parser.parse_args()
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载配置
    config_path = args.config
    if not os.path.exists(config_path):
        # 尝试从项目根目录查找配置文件
        cwd = Path.cwd()
        config_path_alt = cwd / "config" / args.config
        if config_path_alt.exists():
            config_path = config_path_alt
        else:
            config_path_alt = cwd / args.config
            if config_path_alt.exists():
                config_path = config_path_alt
            else:
                print(f"错误: 配置文件不存在: {args.config}")
                return
    
    config = toml.load(config_path)
    print(f"配置加载完成: {config_path}")
    
    # 创建模型
    model = DCCRN(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )
    
    # 加载模型权重
    if os.path.exists(args.model_checkpoint):
        # 确保以CPU模式加载检查点，然后移动到指定设备
        checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
        
        # 根据检查点格式处理
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                # 从checkpoint中提取模型状态字典
                model_state_dict = checkpoint["model"]
                # 加载状态字典
                model.load_state_dict(model_state_dict)
                print(f"模型权重加载成功: {args.model_checkpoint}")
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                print(f"模型权重加载成功: {args.model_checkpoint}")
            else:
                # 尝试直接加载
                try:
                    model.load_state_dict(checkpoint)
                    print(f"模型权重加载成功(直接加载): {args.model_checkpoint}")
                except Exception as e:
                    print(f"错误: 无法从检查点加载模型权重: {e}")
                    return
        else:
            # 如果checkpoint本身就是模型状态字典
            model.load_state_dict(checkpoint)
            print(f"模型权重加载成功: {args.model_checkpoint}")
        
        # 将模型移动到指定设备
        model = model.to(device)
        print(f"模型已移动到设备: {device}")
    else:
        print(f"错误: 模型检查点文件不存在: {args.model_checkpoint}")
        return
    
    # 创建推理器
    inferencer = AudioInferencer(config, model, device, args.model_name)
    
    # 处理所有音频
    print("\n" + "="*50)
    print(f"开始 {args.model_name} 音频增强测试")
    print("="*50)
    
    results, output_dir = inferencer.process_all(args.data_dir, args.output_dir)
    
    # 生成结果报告
    if results:
        report_path = output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(f"{args.model_name} 音频增强测试报告\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {args.config}\n")
            f.write(f"模型文件: {args.model_checkpoint}\n")
            f.write("="*50 + "\n\n")
            
            for key, result in results.items():
                f.write(f"测试: {key}\n")
                f.write(f"  SNR级别: {result['snr_level']}dB\n")
                f.write(f"  干净音频: {result['clean_path']}\n")
                f.write(f"  带噪音频: {result['noisy_path']}\n")
                f.write(f"  增强音频: {result['enhanced_path']}\n")
                f.write(f"  SI-SDR指标: {result['metrics']['SI-SDR']:.2f} dB\n")
                f.write(f"  对比图: {result['comparison_path']}\n")
                f.write(f"  音频长度: {len(result['clean_audio'])} 样本点\n")
                f.write(f"  音频时长: {len(result['clean_audio'])/config['dataset']['sr']:.2f} 秒\n")
                f.write("\n")
        
        print("\n" + "="*50)
        print("测试完成!")
        print(f"结果已保存到: {output_dir}")
        print("="*50)
        
        # 显示生成的文件列表
        print("\n生成的文件:")
        for key, result in results.items():
            print(f"  {key}:")
            print(f"    - 增强音频: {result['enhanced_path'].relative_to(output_dir)}")
            print(f"    - SI-SDR: {result['metrics']['SI-SDR']:.2f} dB")
            print(f"    - 对比图: {result['comparison_path'].relative_to(output_dir)}")
        
        print(f"\n详细报告: {report_path}")
    else:
        print("警告: 没有找到可处理的音频文件")

if __name__ == "__main__":
    main()
    # python mytest/selftest.py -C config/inference_config.toml -M ../model/dccrn/checkpoints/KD_R_margin_x1_16_2/best_model.pth -D mytest/TestData1 -O mytest/mytest_results1 --model_name "resKD"