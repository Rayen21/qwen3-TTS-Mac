#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS FastAPI App (MPS Optimized | HTML Frontend)
✅ 保留原有所有功能逻辑
✅ 改为 FastAPI + HTML 前端
✅ 支持三种模式：官方角色/语音设计/零样本克隆
✅ 优化：仅 Pro-Clone 模式支持 speed 参数
"""
import os
# === 1. 环境配置 (MPS 优化 + 警告抑制) ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
import sys
import gc
import random
import tempfile
import time
import hashlib
import numpy as np
from datetime import datetime
from huggingface_hub import snapshot_download
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import mlx.core as mx
# 关键导入
try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio as generate_audio_func
    import mlx_whisper
except ImportError as e:
    print(f"❌ 缺少依赖：{e}")
    print("请运行：pip install -r requirements.txt")
    sys.exit(1)
# === 2. 全局配置 ===
PROJECT_ROOT = os.getcwd()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
# 模型映射
MODEL_MAP = {
    "Pro-Custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "Pro-Design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "Pro-Clone": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
}
# 角色映射
SPEAKER_MAP = {
    "Chinese": ["Vivian 十三 (活泼女声)", "Serena 苏瑶 (温柔女声)", "Uncle_Fu 福伯 (成熟男声)", "Dylan 晓东 (北京话)", "Eric 程川 (四川话)"],
    "English": ["Aiden 艾登 (自然男声)", "Ryan 甜茶 (年轻男声)", "Serena 苏瑶 (温柔女声)", "Vivian 十三 (活泼女声)"],
    "Japanese": ["Ono_Anna 小野杏 (元气女声)"],
    "Korean": ["Sohee 素熙 (甜美女声)"],
}
SPEAKER_ID_MAP = {
    "Aiden 艾登 (自然男声)": "aiden",
    "Ryan 甜茶 (年轻男声)": "ryan",
    "Vivian 十三 (活泼女声)": "vivian",
    "Serena 苏瑶 (温柔女声)": "serena",
    "Uncle_Fu 福伯 (成熟男声)": "uncle_fu",
    "Dylan 晓东 (北京话)": "dylan",
    "Eric 程川 (四川话)": "eric",
    "Ono_Anna 小野杏 (元气女声)": "ono_anna",
    "Sohee 素熙 (甜美女声)": "sohee"
}
LANGUAGE_CODE_MAP = {
    "Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it"
}
EMOTIONS = []
LANGUAGE_CHOICES = list(SPEAKER_MAP.keys())
# === 3. 全局状态 ===
_model_cache = {}
_current_mode = "Pro-Custom"
# ✅ 新增：种子生成记录缓存（避免重复生成，5 分钟有效期）
_seed_cache = {}  # 格式：{seed: {"params_hash": xxx, "timestamp": xxx, "output_path": xxx}}
# === 4. FastAPI 应用 ===
app = FastAPI(title="VoxCPM Studio Pro")
# === 5. 核心功能函数 (保持原逻辑不变) ===
def _clear_mps_cache():
    """MPS 专用内存清理"""
    try:
        if mx.metal.is_available():
            if hasattr(mx, 'clear_cache'):
                mx.clear_cache()
            elif hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
    except:
        pass
    gc.collect()

def _detect_language(text: str) -> str:
    """自动检测文本语言：增强型 10 语种检测"""
    if not text:
        return "en"
    text_content = text.strip()
    total = len(text_content)
    if total == 0:
        return "en"
    kana_count = sum(1 for c in text_content if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    zh_count = sum(1 for c in text_content if '\u4e00' <= c <= '\u9fff')
    ko_count = sum(1 for c in text_content if '\uac00' <= c <= '\ud7af')
    ru_count = sum(1 for c in text_content if '\u0400' <= c <= '\u04ff')
    if kana_count > 0:
        if kana_count / total > 0.03 or (kana_count + zh_count) / total > 0.3:
            return "ja"
    if ko_count / total > 0.1:
        return "ko"
    if ru_count / total > 0.1:
        return "ru"
    if zh_count / total > 0.2:
        return "zh"
    text_lower = text_content.lower()
    if any(c in text_content for c in 'äöüßÄÖÜ'):
        return "de"
    if any(c in text_content for c in 'ñ¿¡'):
        return "es"
    if any(c in text_content for c in 'ãõÃÕ'):
        return "pt"
    latin_chars = {
        "fr": 'éàèâêîôûëïç',
        "it": 'òì',
        "es": 'áíóú',
    }
    counts = {lang: sum(text_content.count(c) for c in chars) for lang, chars in latin_chars.items()}
    if sum(counts.values()) > 0:
        return max(counts, key=counts.get)
    return "en"

def _get_model(model_key: str):
    """懒加载模型"""
    global _current_mode
    if model_key not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"未知模型：{model_key}")
    folder_name = MODEL_MAP[model_key]
    model_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(model_path):
        snapshots = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots):
            subs = [d for d in os.listdir(snapshots) if not d.startswith('.')]
            if subs:
                model_path = os.path.join(snapshots, subs[0])
            else:
                raise HTTPException(status_code=400, detail=f"模型目录为空：{folder_name}")
        else:
            raise HTTPException(status_code=400, detail=f"模型不存在：{folder_name}")
    if model_key in _model_cache and _current_mode == model_key:
        return _model_cache[model_key]
    if _model_cache:
        print(f"🔄 切换模型：{_current_mode} → {model_key}")
        _model_cache.clear()
        _clear_mps_cache()
    print(f"⏳ 加载模型：{folder_name} ...")
    start = time.time()
    try:
        model = load_model(model_path)
        _model_cache[model_key] = model
        _current_mode = model_key
        print(f"✅ 模型加载完成 ({time.time()-start:.1f}s)")
        return model
    except Exception as e:
        _clear_mps_cache()
        raise HTTPException(status_code=500, detail=f"模型加载失败：{str(e)}")

def _transcribe_audio(audio_path: str) -> str:
    """使用 Whisper 识别文本"""
    if not audio_path or not os.path.exists(audio_path):
        return ""
        
    model_dir = os.path.join(MODELS_DIR, "whisper-large-v3-turbo-asr-fp16")
    if not os.path.exists(model_dir):
        print("⚠️ 请手动下载whisper-large-v3-turbo-asr-fp16到项目models目录")
        return ""
    # 取消自动下载到缓存目录，改成提示用户手动下载到项目models目录下
        #result = generate_transcription(
        #    model="mlx-community/whisper-large-v3-turbo-asr-fp16",
        #    audio=audio_path,  
        #    language=None  
        #)
        #text = result.text.strip()
        #_clear_mps_cache()
        #return text
    # 取消自动下载到缓存目录，改成提示用户手动下载到项目models目录下    
    try:
        print(f"🎙️ Whisper 正在自动识别：{os.path.basename(audio_path)}")
        from mlx_audio.stt.generate import generate_transcription
        return generate_transcription(model=model_dir, audio=audio_path, language=None).text.strip()   
    except Exception as e:
        print(f"⚠️ Whisper 识别失败：{e}")
        _clear_mps_cache()
        return ""
    # 取消自动下载到缓存目录，改成提示用户手动下载到项目models目录下

# ✅ 新增：文本分段函数（放在全局，避免在 if 块内定义）
def split_text_for_tts(text: str, max_chars: int = 80) -> list:
    """按句子/标点智能分割文本，保持语义完整（避免长文本语速漂移）"""
    if len(text) <= max_chars:
        return [text]
    segments = []
    current = ""
    for char in text:
        current += char
        if char in '。！？!?；;.':
            if len(current.strip()) > 0:
                segments.append(current.strip())
                current = ""
            if len(current) > max_chars * 0.8:
                if current.strip():
                    segments.append(current.strip() + "…")
                current = ""
    if current.strip():
        segments.append(current.strip())
    return [s for s in segments if s]

# === 辅助函数：计算文件内容哈希 ===     
def _hash_file_content(file_path: str) -> str:
    """计算文件内容的 MD5，空路径返回空字符串"""
    if not file_path or not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def _generate_tts(text: str, mode: str, language: str = "Chinese", speaker: str = "", emotion: str = "",
                  speed: float = 1.0, ref_audio: str = "", ref_text: str = "",
                  seed: int = -1, design_text: str = ""):
    """TTS 生成主逻辑"""
    model = None
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="⚠️ 合成文本不能为空")
    try:
        # 处理种子：-1 表示随机
        actual_seed = int(seed) if (seed is not None and seed != -1) else random.randint(0, 2**32-1)
        
        # 种子重复检测（5 分钟缓存）- 使用内容哈希替代文件路径
        audio_hash = _hash_file_content(ref_audio) # 调用全局辅助函数
        params_hash = hashlib.md5(
            f"{text.strip()}|{mode}|{language}|{speaker}|{speed}|{audio_hash}".encode()
        ).hexdigest()
        
        if actual_seed in _seed_cache:
            cache_entry = _seed_cache[actual_seed]
            if cache_entry["params_hash"] == params_hash:
                if time.time() - cache_entry["timestamp"] < 300:  # 5 分钟缓存
                    if "output_path" in cache_entry and os.path.exists(cache_entry["output_path"]):
                        print(f"⚡ 检测到相同种子 + 相同参数，使用缓存结果 (Seed: {actual_seed})")
                        return cache_entry["output_path"], actual_seed
        
        mx.random.seed(actual_seed)
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        
        # 2. 加载模型
        model = _get_model(mode)
        start_time = time.time()
        
        # 3.语言判断逻辑（按模式区分）
        if mode == "Pro-Custom":
            if language and language in LANGUAGE_CODE_MAP:
                full_lang = language
                raw_lang = LANGUAGE_CODE_MAP[language]
                print(f"🔤 使用前端指定语言：{full_lang} → {raw_lang}")
            else:
                raw_lang = _detect_language(text)
                full_lang = "Chinese" if raw_lang == "zh" else "English"
                print(f"🔍 自动检测语言（降级）: {raw_lang} → {full_lang}")
            lang_code = raw_lang
        else:
            raw_lang = _detect_language(text)
            REVERSE_MAP = {v: k for k, v in LANGUAGE_CODE_MAP.items()}
            full_lang = REVERSE_MAP.get(raw_lang, "Chinese")
            print(f"🔍 [{mode}] 模式自动语种识别：检测到 '{raw_lang}' ({full_lang})")
            lang_code = raw_lang
        
        # === 调试日志 ===
        print(f"\n🎯 生成参数检查:")
        print(f"   • mode: {mode}")
        print(f"   • speed: {speed} (类型：{type(speed).__name__})")
        print(f"   • full_lang: {full_lang}")
        print(f"   • lang_code: {lang_code}")
        print(f"   • text 长度：{len(text.strip())} chars")
        print(f"   • seed: {actual_seed}")
        
        if not (0.5 <= speed <= 2.0):
            print(f"⚠️ 警告：speed={speed} 超出推荐范围 [0.5, 2.0]，可能被截断或忽略")
        # === 调试日志结束 ===
        
        results = []
        
        # ============================================
        # 🔹 模式 1: Pro-Clone (零样本克隆) - 分段生成防漂移
        # ============================================
        if mode == "Pro-Clone":
            actual_ref_text = ref_text
            if not actual_ref_text or not actual_ref_text.strip():
                actual_ref_text = _transcribe_audio(ref_audio)
            
            print(f"🧬 执行克隆生成 (Seed: {actual_seed}, Speed: {speed}x, Lang: {lang_code})...")
            print(f"   • ref_audio: {ref_audio}")
            print(f"   • ref_text: {actual_ref_text[:50]}..." if actual_ref_text else "   • ref_text: (自动识别)")
            print(f"   • speed 参数值：{speed} (类型：{type(speed).__name__})")
            
            # ✅ 长文本分段处理（避免累积误差导致语速漂移）
            text_segments = split_text_for_tts(text.strip(), max_chars=80)
            print(f"   • 文本分割：{len(text_segments)} 段")
            
            # 分别生成每段音频
            all_audio_chunks = []
            for i, seg_text in enumerate(text_segments):
                print(f"   • 生成第 {i+1}/{len(text_segments)} 段：{seg_text[:20]}...")
                
                # 每段使用相同的 seed + speed，但略微错开避免相位干扰
                seg_seed = actual_seed + i * 1000
                mx.random.seed(seg_seed)
                random.seed(seg_seed)
                np.random.seed(seg_seed)
                
                # 生成单段音频
                seg_results = list(model.generate(
                    text=seg_text,
                    ref_audio=ref_audio,
                    ref_text=actual_ref_text,
                    lang_code=lang_code,
                    speed=speed,
                    verbose=False
                ))
                
                if seg_results and hasattr(seg_results[0], 'audio'):
                    all_audio_chunks.append(np.array(seg_results[0].audio))
            
            # 拼接所有音频段
            if not all_audio_chunks:
                raise Exception("分段生成未返回有效音频")
            
            # 简单拼接音频
            audio_data = np.concatenate(all_audio_chunks)
            
            # 创建兼容的 results 对象
            class Result:
                def __init__(self, audio):
                    self.audio = audio
            results = [Result(mx.array(audio_data))]
            print(f"   • ✅ 分段拼接完成，总时长：{len(audio_data)/24000:.2f}s")
        
        # ============================================
        # 🔹 模式 2: Pro-Design (语音设计) - 使用 VoiceDesign 模型
        # ============================================
        elif mode == "Pro-Design":
            print(f"🎨 执行语音设计生成 (Seed: {actual_seed}, Lang: {full_lang})...")
            results = list(model.generate_voice_design(
                text=text.strip(),
                language=full_lang,
                instruct=design_text or "A natural clear voice."
            ))
            print(f"   • ✅ 语音设计生成完成")
        
        # ============================================
        # 🔹 模式 3: Pro-Custom (官方角色) - 使用 CustomVoice 模型
        # ============================================
        elif mode == "Pro-Custom":
            print(f"👤 执行角色定制生成 (Seed: {actual_seed})...")
            speaker_id = SPEAKER_ID_MAP.get(speaker, speaker.lower() if speaker else "vivian")
            print(f"   [映射转换]: {speaker} -> {speaker_id}")
            
            emotion_instruction = emotion.strip() if emotion else "Natural and clear tone"
            if not emotion_instruction:
                emotion_instruction = "Natural and clear tone"
            
            results = list(model.generate_custom_voice(
                text=text.strip(),
                speaker=speaker_id,
                language=full_lang,
                instruct=emotion_instruction
            ))
            print(f"   • ✅ 角色定制生成完成")
        
        # ============================================
        # 🔹 无效模式处理
        # ============================================
        else:
            raise HTTPException(status_code=400, detail=f"⚠️ 未知模式：{mode}")
        
        # ============================================
        # 🔹 结果验证与保存
        # ============================================
        if not results or not hasattr(results[0], 'audio'):
            raise Exception("模型未返回有效的音频数据")
        
        end_time = time.time()
        elapsed = end_time - start_time
        audio_data = np.array(results[0].audio)
        duration = len(audio_data) / 24000
        print("="*20)
        print(f"Duration:          {duration:.2f}s")
        print(f"Processing Time:   {elapsed:.2f}s")
        print(f"Real-time Factor:  {duration/elapsed:.2f}x")
        try:
            peak_mem = mx.get_peak_memory() / 1024**3
        except AttributeError:
            peak_mem = mx.metal.get_peak_memory() / 1024**3
        print(f"Peak Memory:       {peak_mem:.2f}GB")
        print("="*20)
        
        final_path = os.path.join(tempfile.gettempdir(), f"qwen3_output_{int(time.time())}.wav")
        wavfile.write(final_path, 24000, audio_data)
        
        # ✅ 缓存种子信息（5 分钟有效期）
        _seed_cache[actual_seed] = {
            "params_hash": params_hash,
            "timestamp": time.time(),
            "output_path": final_path
        }
        
        return final_path, actual_seed
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"合成失败：{str(e)}")
    finally:
        _clear_mps_cache()

# === 6. API 路由 ===
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 HTML 前端页面"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
async def get_config():
    """获取前端配置数据"""
    return {
        "languages": LANGUAGE_CHOICES,
        "speakers": SPEAKER_MAP,
        "emotions": [],
        "modes": ["Pro-Custom", "Pro-Design", "Pro-Clone"]
    }

@app.post("/api/transcribe")
async def transcribe_audio(reference_audio: UploadFile = File(...)):
    """音频转文字"""
    try:
        temp_path = os.path.join(tempfile.gettempdir(), f"ref_{int(time.time())}_{reference_audio.filename}")
        with open(temp_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)
        text = _transcribe_audio(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败：{str(e)}")

@app.post("/api/generate")
async def generate_audio(
    text: str = Form(...),
    mode: str = Form("Pro-Custom"),
    language: str = Form("Chinese"),
    speaker: str = Form(""),
    emotion: str = Form(""),
    speed: float = Form(1.0),
    seed: int = Form(-1),
    design_text: str = Form(""),
    prompt_text: str = Form(""),
    reference_audio: UploadFile = File(None)
):
    """生成 TTS 音频"""
    temp_audio_path = None
    try:
        if reference_audio and reference_audio.filename:
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"ref_{int(time.time())}_{reference_audio.filename}")
            with open(temp_audio_path, "wb") as f:
                content = await reference_audio.read()
                f.write(content)
        
        output_path, actual_seed = _generate_tts(
            text=text,
            mode=mode,
            language=language,
            speaker=speaker,
            emotion=emotion,
            speed=speed,
            ref_audio=temp_audio_path if temp_audio_path else "",
            ref_text=prompt_text,
            seed=seed,
            design_text=design_text
        )
        
        response = FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"output_{actual_seed}.wav"
        )
        response.headers["X-Actual-Seed"] = str(actual_seed)
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# === 7. 启动入口 ===
if __name__ == "__main__":
    print("🚀 Qwen3TTS 服务启动中 (MPS Optimized)...")
    print(f"📁 模型目录：{MODELS_DIR}")
    print(f"🖥️  MPS 可用：{mx.metal.is_available()}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9860,
        log_level="info"
    )
