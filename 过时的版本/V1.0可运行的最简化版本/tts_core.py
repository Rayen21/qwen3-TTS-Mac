# tts_core.py
import os
import time
import tempfile
import shutil
import gc

from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

# ===== 项目目录结构 =====
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 选择默认模型（你可以以后做成 UI 下拉框）=====
HF_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
LOCAL_MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

# ===== 全局模型缓存 =====
_model = None


def _ensure_model_downloaded():
    """
    如果模型不在本地，就从 HuggingFace 下载到 ./models
    """
    local_path = os.path.join(MODELS_DIR, LOCAL_MODEL_NAME)

    if os.path.exists(local_path):
        return _resolve_snapshot_path(local_path)

    print(f"[INFO] Downloading model to {local_path} ...")

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    return _resolve_snapshot_path(local_path)


def _resolve_snapshot_path(model_dir):
    """
    兼容 HF 的 snapshots 目录结构
    """
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshots):
        subs = [f for f in os.listdir(snapshots) if not f.startswith(".")]
        if subs:
            return os.path.join(snapshots, subs[0])
    return model_dir


def _get_model():
    global _model
    if _model is None:
        model_path = _ensure_model_downloaded()
        _model = load_model(model_path)
    return _model


def tts(
    text: str,
    speaker: str = "Vivian",
    emotion: str = "Normal tone",
    speed: float = 1.0,
) -> str:
    if not text.strip():
        raise ValueError("Text is empty")

    model = _get_model()

    temp_dir = tempfile.mkdtemp(prefix="tts_")

    try:
        generate_audio(
            model=model,
            text=text,
            voice=speaker,
            instruct=emotion,
            speed=speed,
            output_path=temp_dir,
        )

        src = os.path.join(temp_dir, "audio_000.wav")
        if not os.path.exists(src):
            raise RuntimeError("Audio generation failed")

        final = os.path.join(
            tempfile.gettempdir(),
            f"qwen3_tts_{int(time.time())}.wav",
        )
        shutil.move(src, final)
        return final

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()
