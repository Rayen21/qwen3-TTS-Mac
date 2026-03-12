# qwen3-TTS-Mac
## qwen3-TTS适用于Mac的版本

### 使用mlx-community模型，MPS加速，生成速度快，占用内存小，一键启动（qwen3-tts.sh脚本）

### 20260311功能更新：
1、美化UI界面（抛弃gradio使用index）

2、零样本克隆和语音设计页，根据输入文本正确进行多语种识别："Chinese":"zh""English":"en","Japanese":"ja","Korean":"ko"，"French":"fr","German":"de","Spanish":"es","RuSsian":"ru","Portuguese":"pt","Italian":"it"

3、对长句进行了分段拆分，避免零样本克隆时，长句生成音频会导致语速越来越快的问题。

4、语速调节功能测试下来没有效果，在代码里已经给隐藏了（3中已经解决了生成长句的语速越来越快问题，所有个人觉的也没必要加语速控制了）

5、点击随机种子即刻会重新生成，不用再点击一次种子再点击一次生成，简化了操作（默认设定是上次生成的种子数一直保留，直到用户点击随机种子才产生新的随机种子）

### 新界面
<img width="1530" height="981" alt="image" src="https://github.com/user-attachments/assets/27f8b435-147d-4758-be19-6a70958e8806" />

<img width="1529" height="981" alt="image" src="https://github.com/user-attachments/assets/03e9e75a-f59d-4b07-9c9f-0d332f872650" />

<img width="1529" height="981" alt="image" src="https://github.com/user-attachments/assets/baa8b652-6bed-4a3f-89fe-28e0a1ba6128" />

### 运行日志
```
🔍 [Pro-Clone] 模式自动语种识别：检测到 'zh' (Chinese)

🎯 生成参数检查:
   • mode: Pro-Clone
   • speed: 1.0 (类型：float)
   • full_lang: Chinese
   • lang_code: zh
   • text 长度：171 chars
   • seed: 1823475686
🧬 执行克隆生成 (Seed: 1823475686, Speed: 1.0x, Lang: zh)...
   • ref_audio: /var/folders/g2/9nr8q_kj2tggghtk37whpbsh0000gn/T/ref_1773296826_呃……朋友们好啊，我是浑元形意太极门掌门，
     马保国。刚才有个朋友问我，马老师发生甚么事了？我是怎么回事？.wav
   • ref_text: 呃……朋友们好啊，我是浑元形意太极门掌门，马保国。刚才有个朋友问我，马老师发生甚么事了？我是怎么回事...
   • speed 参数值：1.0 (类型：float)
   • 文本分割：5 段
   • 生成第 1/5 段：你迈开脚步，试图穿过这间屋子，然而你的鞋...
   • 生成第 2/5 段：墙壁仿佛有了呼吸，随着你的靠近而微微起伏...
   • 生成第 3/5 段：你走了很久，久到忘记了时间的流逝。...
   • 生成第 4/5 段：起初的几步，走廊明明短得可一眼望穿，可不...
   • 生成第 5/5 段：你的影子在墙上被拉长、扭曲，时而变成你，...
   • ✅ 分段拼接完成，总时长：33.20s
====================
Duration:          33.20s
Processing Time:   16.22s
Real-time Factor:  2.05x
Peak Memory:       7.69GB
====================
INFO:     127.0.0.1:55602 - "POST /api/generate HTTP/1.1" 200 OK
```

## 安装注意事项简述：

1、clone项目到本地后，使用conda建立环境

2、使用mlx-community的qwen3tts模型生成音频和mlx_whisper的模型来识别参考音转换文本

模型放置地址在项目目录下 
```
～/models/Qwen3-TTS-12Hz-1.7B-Base-8bit
～/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
～/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
～/models/mlx_whisper/whisper-large-v3-turbo-asr-fp16
```

3、如果要支持上传多种音频格式（m4a,mp3等）需要安装ffmpeg，否则会产生错误，请自己在conda激活qwen3-tts项目环境后，自行安装一下。
```
conda install -c conda-forge ffmpeg
```
4、模型没有设定成自动下载，需要自己对应下载相关模型，因为本项目初衷是为了替代index-tts的“克隆”功能，所以怕使用的时候点到别的模式触发自动下载模型，占用硬盘空间。

5、qwen3-tts.sh的使用，请自行"chmod +x qwen3-tts.sh"在终端中赋予执行权限。

6、本项目只是个人爱好，并不会定期更新功能。

### 感谢这个作者写的qwen3tts-mlx-audio项目，我在他的基础上修改而来

https://github.com/kapi2800/qwen3-tts-apple-silicon?tab=readme-ov-file

### 模型下载地址 (默认使用1.7B-8bit的模型(260228))
##### <i>~~合并代码为单文件版后已经是混合两种模型使用了~~</i>
```
模型分类：#官方预设角色 CustomVoice #语音设计 VoiceDesign #零样本克隆 Base 
Pro Models (1.7B) - Best Quality
CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
Lite Models (0.6B) - Faster, Less RAM 
CustomVoice https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
VoiceDesign https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit
Base        https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
whisper     https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16
```
<h3 align="center" style="font-size: 48px; color: #3399ff; font-style: italic;">
  ------------------------以下为旧版（可以不用看了）------------------------
</h3>

##### #预设角色音频调用功能
<img width="1024" height="435" alt="1" src="https://github.com/user-attachments/assets/94c36d11-b9c5-4e11-8235-38e5ef3be95d" />

##### #自主语音设计功能
<img width="1024" height="435" alt="2" src="https://github.com/user-attachments/assets/e766c722-3b8a-47fa-8d02-c1b618545ddf" />

##### #零样本语音克隆功能
<img width="1024" height="435" alt="3" src="https://github.com/user-attachments/assets/e5bd5b57-c182-4575-be80-031ffe14a65e" />

##### #生成速度及资源占用


<img width="293" height="264" alt="截屏2026-02-28 18 21 48" src="https://github.com/user-attachments/assets/aa9bb79e-4612-4bfb-8529-c0000e3a3582" />


