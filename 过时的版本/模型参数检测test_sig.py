import sys, os
sys.path.insert(0, os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mlx_audio.tts.utils import load_model
import inspect

# 加载一个已下载的模型（请替换为您的实际模型路径）
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"  # 请确认路径存在
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-0.6B-Base-bf16"
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
#model_path = "/Users/hanqingren/qwen3tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
if os.path.exists(model_path):
    model = load_model(model_path)
    if hasattr(model, 'generate'):
        sig = inspect.signature(model.generate)
        print("✅ model.generate 参数:")
        for n, p in sig.parameters.items():
            print(f"  {n}: {p.annotation if p.annotation != inspect.Parameter.empty else 'Any'} = {p.default if p.default != inspect.Parameter.empty else 'REQUIRED'}")
    else:
        print("❌ model 没有 generate 方法")
        print("可用方法:", [m for m in dir(model) if not m.startswith('_')])
else:
    print(f"⚠️ 模型路径不存在: {model_path}")
