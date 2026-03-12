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
# 记录当前加载的模型 Key，用于判断是否需要清理内存并切换
_current_loaded_key = None
# 默认模式
LOCAL_MODEL_NAME = "Lite-Custom"

# 严格对应文件夹名称的映射表
MODEL_MAP = {
    "Pro-Custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "Pro-Design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "Pro-Clone": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "Lite-Custom": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "Lite-Design": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit",
    "Lite-Clone": "Qwen3-TTS-12Hz-0.6B-Base-8bit",
}

def _get_model(model_key):
    """
    智能模型加载器：支持自动清理旧模型并释放显存
    """
    global _model, _current_loaded_key
    
    folder_name = MODEL_MAP.get(model_key)
    if not folder_name:
        raise gr.Error(f"未知的模型 Key: {model_key}")
        
    local_path = os.path.join(MODELS_DIR, folder_name)

    # 改进：仅在模型发生切换时才清理显存并重新加载
    if _model is not None and _current_loaded_key != model_key:
        print(f"[CLEANUP] 检测到模型切换 ({_current_loaded_key} -> {model_key})，正在释放内存...")
        del _model
        _model = None
        gc.collect()

    if _model is None:
        print(f"[INFO] 正在加载模型: {folder_name}")
        if not os.path.exists(local_path):
            raise gr.Error(f"模型文件不存在，请检查目录: {folder_name}")
        _model = load_model(local_path)
        _current_loaded_key = model_key
        
    return _model

def transcribe_audio(audio_path):
    """
    使用 Whisper 自动识别参考音频内容
    """
    if not audio_path:
        return ""
    
    print(f"[INFO] 正在自动识别参考音频...")
    try:
        # 加载 Whisper base 模型（轻量且适合中文）
        stt_model = whisper.load_model("base")
        result = stt_model.transcribe(audio_path, initial_prompt="以下是普通话。")
        identified_text = result["text"].strip()
        print(f"[INFO] 识别完成: {identified_text}")
        return identified_text
    except Exception as e:
        print(f"[ERROR] Whisper 识别失败: {e}")
        return "【识别失败】请手动输入参考文本"

def tts_all_in_one(text, speaker, instruct, speed, ref_audio, ref_text, model_size, seed=-1):
    """
    TTS 主入口：处理种子、模型加载及音频合成
    """
    global LOCAL_MODEL_NAME
    
    # 确定模型规格
    size = model_size if model_size else "Lite"
    mode_suffix = LOCAL_MODEL_NAME.split("-")[-1]
    full_model_key = f"{size}-{mode_suffix}"
    
    # 种子锁定逻辑
    actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    mx.random.seed(actual_seed)
    print(f"[DEBUG] 使用种子: {actual_seed} | 模式: {full_model_key}")

    try:
        model = _get_model(full_model_key)
    except Exception as e:
        raise gr.Error(str(e))

    temp_dir = tempfile.mkdtemp()
    
    # 零样本克隆模式下，如果没填文字则自动识别
    actual_ref_text = ref_text
    if full_model_key.endswith("Clone") and ref_audio and not ref_text.strip():
        actual_ref_text = transcribe_audio(ref_audio)

    try:
        print(f"[INFO] 正在生成语音，模式: {full_model_key}")
        generate_audio(
            model=model,
            text=text,
            voice=speaker.lower() if speaker else None,
            instruct=instruct,
            speed=speed,
            ref_audio=ref_audio,
            ref_text=actual_ref_text,
            output_path=temp_dir,
            language="zh"
        )
        
        src = os.path.join(temp_dir, "audio_000.wav")
        if not os.path.exists(src):
            raise RuntimeError("音频生成失败")

        final_path = os.path.join(tempfile.gettempdir(), f"qwen3_out_{int(time.time())}.wav")
        shutil.copy(src, final_path)
        return final_path, actual_ref_text, actual_seed

    except Exception as gen_e:
        print(f"[ERROR] 合成失败: {gen_e}")
        raise gr.Error(f"合成失败: {gen_e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        gc.collect()