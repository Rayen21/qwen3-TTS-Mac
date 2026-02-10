# tts_core.py
import os
import time
import tempfile
import shutil
import random
import numpy as np
import gc
import gradio as gr
import mlx.core as mx
import whisper

from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

# --- 路径与全局变量配置 ---
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 全局模型缓存
_model = None
_current_loaded_key = None
# 默认模式锁定为 Pro
LOCAL_MODEL_NAME = "Pro-Custom"

# 清理后的模型映射表：仅保留 Pro (1.7B) 版本
MODEL_MAP = {
    "Pro-Custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "Pro-Design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "Pro-Clone": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
}

def _get_model(model_key):
    """锁定加载 1.7B Pro 规格模型"""
    global _model, _current_loaded_key
    
    folder_name = MODEL_MAP.get(model_key)
    if not folder_name:
        raise gr.Error(f"未知的 Pro 模型 Key: {model_key}")
        
    local_path = os.path.join(MODELS_DIR, folder_name)

    if _model is not None and _current_loaded_key != model_key:
        print(f"[CLEANUP] 模型切换 ({_current_loaded_key} -> {model_key})，释放内存...")
        del _model
        _model = None
        gc.collect()

    if _model is None:
        if not os.path.exists(local_path):
            raise gr.Error(f"Pro 模型文件不存在: {folder_name}")
        _model = load_model(local_path)
        _current_loaded_key = model_key
        
    return _model

def transcribe_audio(audio_path):
    """Whisper 自动识别"""
    if not audio_path: return ""
    try:
        stt_model = whisper.load_model("base")
        result = stt_model.transcribe(audio_path, initial_prompt="以下是普通话。")
        return result["text"].strip()
    except Exception as e:
        return "【识别失败】"

def tts_all_in_one(text, speaker, emotion, speed, ref_audio, ref_text, seed=-1):
    """
    清理后的 TTS 入口：强制使用 Pro 逻辑
    """
    global LOCAL_MODEL_NAME
    # 强制锁定为 Pro 规格
    full_model_key = LOCAL_MODEL_NAME 
    
    actual_seed = int(seed) if (seed is not None and seed != -1) else random.randint(0, 2**32 - 1)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    mx.random.seed(actual_seed)

    model = _get_model(full_model_key)
    temp_dir = tempfile.mkdtemp()
    
    actual_ref_text = ref_text
    if full_model_key.endswith("Clone") and ref_audio and not ref_text.strip():
        actual_ref_text = transcribe_audio(ref_audio)

    try:
        generate_audio(
            model=model,
            text=text,
            voice=speaker.lower() if speaker else None,
            instruct=emotion,
            speed=speed,
            ref_audio=ref_audio,
            ref_text=actual_ref_text,
            output_path=temp_dir,
            language="zh"
        )
        
        src = os.path.join(temp_dir, "audio_000.wav")
        final_path = os.path.join(tempfile.gettempdir(), f"qwen3_pro_{int(time.time())}.wav")
        shutil.copy(src, final_path)
        return final_path, actual_seed

    except Exception as e:
        raise gr.Error(f"Pro 模型合成失败: {e}")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        gc.collect()