#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS Gradio App (MPS Optimized | Single File | Fixed)
✅ 修复: 按钮重复绑定导致双重生成
✅ 修复: voice/ref_audio 参数冲突
✅ 修复: Language 硬编码为 en
✅ 修复: mx.metal.clear_cache 弃用警告
✅ 修复: Gradio theme 参数位置警告
✅ 优化: 中文文本自动检测语言
"""

import os
import sys
import gc
import random
import shutil
import tempfile
import time
import warnings
from datetime import datetime
from huggingface_hub import snapshot_download #检测模型不存在时自动下载

# === 1. 环境配置 (MPS 优化 + 警告抑制) ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["FIX_MISTRAL_REGEX"] = "1"  # 🔧 修复 tokenizer 正则警告

# 抑制已知无害警告
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
warnings.filterwarnings("ignore", message=".*model of type qwen3_tts.*")

import gradio as gr
import mlx.core as mx
import numpy as np

# 关键导入
try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
    import mlx_whisper
except ImportError as e:
    print(f"❌ 缺少依赖: {e}\n请运行: pip install -r requirements.txt")
    sys.exit(1)

# === 2. 全局配置 ===
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 模型映射 
MODEL_MAP = {
    #"Pro-Custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", #官方预设角色
    "Pro-Custom": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "Pro-Design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16", #语音设计
    #"Pro-Clone": "Qwen3-TTS-12Hz-1.7B-Base",              #零样本克隆
    "Pro-Clone": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    #"Pro-Clone": "Qwen3-TTS-12Hz-0.6B-Base-bf16",
}

# UI 配置数据
SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}
EMOTIONS = ["Normal tone", "Sad", "Excited", "Angry", "Whispering"]
LANGUAGE_CHOICES = list(SPEAKER_MAP.keys())

# === 3. 全局状态 ===
_model_cache = {}
_current_mode = "Pro-Custom"

# === 4. 核心功能函数 ===

def _clear_mps_cache():
    """MPS 专用内存清理 - 适配新版 mlx"""
    try:
        if mx.metal.is_available():
            # 🔧 新版 mlx 使用 mx.clear_cache()
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()  # 兼容旧版
    except:
        pass
    gc.collect()

def _detect_language(text: str) -> str:
    """自动检测文本语言 (zh/en)"""
    if not text:
        return "en"
    zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 中文字符占比 >30% 则判定为中文
    return "zh" if zh_count / len(text) > 0.3 else "en"

def _get_model(model_key: str):
    """懒加载模型"""
    global _current_mode
    
    if model_key not in MODEL_MAP:
        raise gr.Error(f"未知模型: {model_key}")
    
    folder_name = MODEL_MAP[model_key]
    model_path = os.path.join(MODELS_DIR, folder_name)
    
    # 兼容 snapshots 目录结构
    if not os.path.exists(model_path):
        snapshots = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots):
            subs = [d for d in os.listdir(snapshots) if not d.startswith('.')]
            if subs:
                model_path = os.path.join(snapshots, subs[0])
            else:
                raise gr.Error(f"模型目录为空: {folder_name}")
        else:
            raise gr.Error(f"模型不存在: {folder_name}")
    
    # 缓存命中
    if model_key in _model_cache and _current_mode == model_key:
        return _model_cache[model_key]
    
    # 切换模型时清理
    if _model_cache:
        print(f"🔄 切换模型: {_current_mode} → {model_key}")
        _model_cache.clear()
        _clear_mps_cache()
    
    print(f"⏳ 加载模型: {folder_name} ...")
    start = time.time()
    
    try:
        model = load_model(model_path)
        _model_cache[model_key] = model
        _current_mode = model_key
        print(f"✅ 模型加载完成 ({time.time()-start:.1f}s)")
        return model
    except Exception as e:
        _clear_mps_cache()
        raise gr.Error(f"模型加载失败: {str(e)}")

def _transcribe_audio(audio_path: str) -> str:
    """使用 mlx-whisper 进行语音识别 (Apple Silicon 优化版)"""
    if not audio_path:
        return ""
       
    # 设定模型存储根目录
    WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "mlx_whisper")
    os.makedirs(WHISPER_MODELS_DIR, exist_ok=True)
    
    # 你想要使用的模型 ID
    model_id = "mlx-community/whisper-base-mlx" 
    
    # 构造该模型的本地特定目录
    # 例如：models/mlx_whisper/whisper-base-mlx
    local_model_path = os.path.join(WHISPER_MODELS_DIR, model_id.split('/')[-1])

    try:
        # 1. 检查并下载模型到指定目录
        if not os.path.exists(local_model_path):
            print(f"⏳ 正在下载 Whisper 模型到本地目录: {local_model_path}...")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_model_path,
                local_dir_use_symlinks=False # 禁用符号链接，确保文件实实在在下载到该目录
            )
            print("✅ 下载完成")

        print(f"🎙️ 正在识别 (使用本地模型): {os.path.basename(audio_path)}")
        
        # 2. 调用时传入本地路径而非 Repo ID
        result = mlx_whisper.transcribe(
            audio_path, 
            path_or_hf_repo=local_model_path 
        )
        
        _clear_mps_cache()
        return result["text"].strip()
        
    except Exception as e:
        print(f"⚠️ Whisper 识别或下载失败: {e}")
        return "【识别失败】"

def _generate_tts(text: str, speaker: str, emotion: str, speed: float, 
                  ref_audio: str, ref_text: str, seed: int, model_key: str):
    """TTS 生成主逻辑 - 已优化以符合 mlx-audio 规范"""
    if not text or not text.strip():
        raise gr.Error("⚠️ 合成文本不能为空")
    
    # 种子处理
    actual_seed = int(seed) if (seed is not None and seed != -1) else random.randint(0, 2**32-1)
    mx.random.seed(actual_seed)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    
    # 加载模型
    model = _get_model(model_key)
    temp_dir = tempfile.mkdtemp(prefix="qwen3_tts_")
    
    try:
        is_clone_mode = "Clone" in model_key
        lang = _detect_language(text)
        
        # 规范化参数调用
        gen_params = {
            "model": model,
            "text": text.strip(),
            "instruct": emotion,
            "speed": speed,
            "output_path": temp_dir,
            "language": lang
        }

        if is_clone_mode and ref_audio:
            # === 符合 mlx-audio 规范的克隆调用 ===
            actual_ref_text = ref_text
            if not actual_ref_text or not actual_ref_text.strip():
                print("🎤 自动识别参考音频...")
                actual_ref_text = _transcribe_audio(ref_audio)
            
            print(f"🧬 克隆模式: ref_audio={os.path.basename(ref_audio)}, lang={lang}")
            gen_params.update({
                "ref_audio": ref_audio,
                "ref_text": actual_ref_text,
                "voice": None  # 明确移除预设音色
            })
        else:
            # === 标准角色模式调用 ===
            voice_name = speaker.lower() if speaker else "vivian"
            print(f"👤 角色模式: voice={voice_name}, lang={lang}")
            gen_params.update({
                "voice": voice_name,
                "ref_audio": None,
                "ref_text": None
            })
        
        # 执行生成
        generate_audio(**gen_params)
        
        # 复制输出文件
        src = os.path.join(temp_dir, "audio_000.wav")
        final_path = os.path.join(tempfile.gettempdir(), f"qwen3_pro_{int(time.time())}.wav")
        shutil.copy(src, final_path)
        return final_path, actual_seed

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Pro 模型合成失败: {e}")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        _clear_mps_cache()

# === 5. Gradio UI 构建 ===

def update_speakers(lang: str):
    """语言切换联动"""
    speakers = SPEAKER_MAP.get(lang, [])
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)

def switch_mode(mode_label: str):
    """模式切换 + 更新全局模型 Key"""
    global _current_mode
    mapping = {
        "官方角色": "Pro-Custom",
        "语音设计": "Pro-Design", 
        "零样本克隆": "Pro-Clone"
    }
    _current_mode = mapping.get(mode_label, "Pro-Custom")
    
    return [
        gr.update(visible=(mode_label == "官方角色")),
        gr.update(visible=(mode_label == "语音设计")),
        gr.update(visible=(mode_label == "零样本克隆"))
    ]

# 🔧 修复: theme 参数移到 launch() 方法
with gr.Blocks(title="Qwen3-TTS Pro") as demo:
    gr.Markdown("## 🎙️ Qwen3 Neural Voice Engine (MPS Optimized)")
    
    with gr.Row():
        # === 左侧控制面板 ===
        with gr.Column(scale=1):
            mode_nav = gr.Radio(
                ["官方角色", "语音设计", "零样本克隆"], 
                label="🔧 功能模式", 
                value="官方角色"
            )
            seed_input = gr.Number(value=-1, label="🎲 随机种子 (-1=随机)", precision=0)
            
            # 模式 1: 官方角色
            with gr.Group(visible=True) as group_custom:
                gr.Markdown("### 👤 角色设置")
                lang_sel = gr.Dropdown(LANGUAGE_CHOICES, value="Chinese", label="语言")
                spk_sel = gr.Dropdown(SPEAKER_MAP["Chinese"], value="Vivian", label="角色")
                emo_sel = gr.Dropdown(EMOTIONS, value="Normal tone", label="情感")
                speed_sel = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="语速")
            
            # 模式 2: 语音设计
            with gr.Group(visible=False) as group_design:
                gr.Markdown("### 🎨 声音描述")
                design_input = gr.Textbox(
                    label="描述提示词", 
                    placeholder="例: 磁性男声，略带沙哑，语速缓慢",
                    lines=2
                )
            
            # 模式 3: 零样本克隆
            with gr.Group(visible=False) as group_clone:
                gr.Markdown("### 🧬 参考音频")
                ref_aud = gr.Audio(label="上传参考音频 (≤30s)", type="filepath")
                ref_txt = gr.Textbox(label="参考文本 (可选，留空自动识别)", lines=2)
                ref_aud.change(
                    fn=_transcribe_audio, 
                    inputs=ref_aud, 
                    outputs=ref_txt,
                    show_progress="minimal"
                )
        
        # === 右侧输出面板 ===
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 合成文本", 
                lines=6, 
                placeholder="输入要合成的内容...",
                max_lines=20
            )
            gen_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
            
            # 🔧 修复: autoplay=False 避免双重播放
            out_aud = gr.Audio(label="🔊 输出结果", interactive=False, autoplay=True)
            res_seed = gr.Number(label="实际使用种子", interactive=False)
            
            gr.HTML("""
                <div style="text-align:center;margin-top:20px;color:#666;font-size:12px">
                    MPS Accelerated | Auto Memory Cleanup | 
                    <a href="https://github.com/Rayen21/qwen3-TTS-Mac" target="_blank">GitHub</a>
                </div>
            """)
    
    # === 事件绑定 ===
    lang_sel.change(fn=update_speakers, inputs=lang_sel, outputs=spk_sel)
    mode_nav.change(
        fn=switch_mode, 
        inputs=mode_nav, 
        outputs=[group_custom, group_design, group_clone]
    )
    
    # 🔧 关键修复: 只绑定一次，使用全局 _current_mode 传递 model_key
    gen_btn.click(
        fn=lambda t, spk, emo, spd, ra, rt, sd: _generate_tts(
            t, spk, emo, spd, ra, rt, sd, _current_mode
        ),
        inputs=[text_input, spk_sel, emo_sel, speed_sel, ref_aud, ref_txt, seed_input],
        outputs=[out_aud, res_seed],
        show_progress="full"
    )

# === 6. 启动入口 ===
if __name__ == "__main__":
    print("🔧 Qwen3-TTS Pro 启动中 (MPS 优化版 | Fixed)...")
    print(f"📁 模型目录: {MODELS_DIR}")
    print(f"🖥️  MPS 可用: {mx.metal.is_available()}")
    
    try:
        # 🔧 修复: theme 参数移到 launch() 方法
        demo.launch(
            server_port=9860,
            inbrowser=True,
            quiet=False,
            theme=gr.themes.Soft()
        )
    except KeyboardInterrupt:
        print("\n👋 正在清理内存...")
        _clear_mps_cache()
        print("✅ 退出完成")
