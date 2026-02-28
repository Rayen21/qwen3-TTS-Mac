# qwen3-TTS-Mac
## qwen3-TTS适用于Mac的版本

### 使用MLX模型，MPS加速，生成速度快，占用内存小，一键启动（qwen3-tts.sh脚本）

##### #预设角色音频调用功能
<img width="1024" height="435" alt="1" src="https://github.com/user-attachments/assets/94c36d11-b9c5-4e11-8235-38e5ef3be95d" />

##### #自主语音设计功能
<img width="1024" height="435" alt="2" src="https://github.com/user-attachments/assets/e766c722-3b8a-47fa-8d02-c1b618545ddf" />

##### #零样本语音克隆功能
<img width="1024" height="435" alt="3" src="https://github.com/user-attachments/assets/e5bd5b57-c182-4575-be80-031ffe14a65e" />

##### #生成速度及资源占用


<img width="293" height="264" alt="截屏2026-02-28 18 21 48" src="https://github.com/user-attachments/assets/aa9bb79e-4612-4bfb-8529-c0000e3a3582" />


## 安装注意事项简述：

1、clone项目到本地后，使用conda建立环境

2、mlx_whisper来识别参考音转换文本（模型地址在～/models/mlx_whisper/whisper-base-mlx）

3、如果没有安装ffmpeg，在生成时会产生错误，请自己在conda激活qwen3-tts项目环境后，自行安装一下。
```python
conda install -c conda-forge ffmpeg
```
4、模型没有设定成自动下载，需要自己对应下载相关模型，因为本项目初衷是为了替代index-tts的“克隆”功能，所以怕使用的时候点到别的模式触发自动下载模型，占用硬盘空间。

5、qwen3-tts.sh的使用，请自行"chmod +x qwen3-tts.sh"在终端中赋予执行权限。

6、本项目只是个人爱好，并不会定期更新功能。


### 模型下载地址 

模型分类：#官方预设角色 CustomVoice #语音设计 VoiceDesign #零样本克隆 Base 

Pro Models (1.7B) - Best Quality

CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit

Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit

Lite Models (0.6B) - Faster, Less RAM ##默认使用了1.7B-8bit的模型(260228)## ~~合并代码为单文件版后已经是混合两种模型使用了~~

CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit

VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit

Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
