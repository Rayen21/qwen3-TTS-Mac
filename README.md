# qwen3-TTS-Mac
## qwen3-TTS适用于Mac的版本

### 使用MLX模型，MPS加速生成速度快，占用内存小，一键启动（qwen3-tts.sh脚本）

<img width="1547" height="1033" alt="image" src="https://github.com/user-attachments/assets/daff7a81-bdb5-4938-aa86-4e6021575187" />
<img width="1547" height="1015" alt="image" src="https://github.com/user-attachments/assets/ee6e3948-a0aa-4974-b066-38e27c1bf629" />
<img width="483" height="108" alt="image" src="https://github.com/user-attachments/assets/494787d7-07a1-4f50-b042-c112dd4d0a71" />

## 安装注意事项简述：

1、clone项目到本地后，使用conda建立环境

2、第一次运行“克隆”模式时，whisper需要下载一个base模型到“～.cache/whisper/base.pt”，如果想要切换tiny模型，请自己在tts_core.py里修改，***whisper自动识别的文本不正确的地方请自己在参考文本框手动修改。***

```python
def transcribe_audio(audio_path):
    """Whisper 自动识别"""
    global _stt_model
    if not audio_path: 
        return ""
    try:
        if _stt_model is None:
            # 第一次加载可能会慢，因为需要下载模型 (约 140MB)
            print("--- 正在初始化 Whisper (Base) 模型 ---")
            _stt_model = whisper.load_model("base")
```
3、如果没有安装ffmpeg，在生成时会产生错误，请自己在conda激活qwen3-tts项目环境后，自行安装一下。
```python
conda install -c conda-forge ffmpeg
```
4、模型没有设定成自动下载，需要自己对应下载相关模型，因为本项目初衷是为了替代index-tts的“克隆”功能，所以怕使用的时候点到别的模式触发自动下载模型，占用硬盘空间。

5、qwen3-tts.sh的使用，请自行"chmod +x qwen3-tts.sh"在终端中赋予执行权限。

6、本项目只是个人爱好改写自下面作者，并不会定期更新功能。

## 感谢这个作者写的主代码

https://github.com/kapi2800/qwen3-tts-apple-silicon?tab=readme-ov-file

我只是基于这个作者的代码作了一个gradio的操作前端。


### 模型下载地址

Pro Models (1.7B) - Best Quality

CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit

Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit

Lite Models (0.6B) - Faster, Less RAM **(因为实测在M2 MacBookair上也能很快速的使用1.7B的模型，所以代码中已经删除了0.6B的部分）**

***强烈建议1.7B模型使用，质量明显优于0.6B，同时速度感觉并没有什么显著差别***

CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit

VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit

Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
