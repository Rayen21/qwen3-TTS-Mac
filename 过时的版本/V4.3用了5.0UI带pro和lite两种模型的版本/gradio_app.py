# gradio_app.py
import gradio as gr
import tts_core
import random
import os

# =========================================================
# 联动配置数据
# =========================================================
SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}
LANGUAGE_CHOICES = list(SPEAKER_MAP.keys())
EMOTION_EXAMPLES = ["Normal tone", "Sad and crying, speaking slowly", "Excited and happy, speaking very fast", "Angry and shouting", "Whispering quietly"]

# =========================================================
# UI 联动逻辑函数
# =========================================================
def update_speakers(lang):
    """根据选择的语言更新对应的说话人列表"""
    speakers = SPEAKER_MAP.get(lang, [])
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)

def switch_ui_mode(mode_label):
    """控制左侧不同模式分组的可见性"""
    mapping = {"官方角色": "Lite-Custom", "语音设计": "Lite-Design", "零样本克隆": "Lite-Clone"}
    tts_core.LOCAL_MODEL_NAME = mapping[mode_label]
    return [
        gr.update(visible=(mode_label == "官方角色")),
        gr.update(visible=(mode_label == "语音设计")),
        gr.update(visible=(mode_label == "零样本克隆"))
    ]

# =========================================================
# 核心验证与生成逻辑 (修复路径与种子)
# =========================================================
def validated_tts(text, mode, model_size, lang, speaker, emotion, design_text, ref_audio, ref_text, seed):
    # 1. 基础校验
    if not text or text.strip() == "":
        gr.Warning("⚠️ 请先输入需要合成的文本")
        return None, -1

    # 2. 种子安全转换
    safe_seed = int(seed) if (seed is not None and seed != -1) else random.randint(0, 2**32 - 1)

    # 3. 精准路径匹配 (修复重复 Base 问题)
    version = "1.7B" if model_size == "Pro" else "0.6B"
    if mode == "官方角色":
        suffix = "CustomVoice"
    elif mode == "语音设计":
        suffix = "VoiceDesign"
    else:
        suffix = "Base" # 对应本地 Qwen3-TTS-12Hz-0.6B-Base-8bit
    
    folder_name = f"Qwen3-TTS-12Hz-{version}-{suffix}-8bit"
    model_path = os.path.join("models", folder_name)

    if not os.path.exists(model_path):
        gr.Warning(f"🚫 模型目录不存在: {folder_name}")
        return None, safe_seed

    try:
        # 调用后端逻辑
        # 注意：这里需要根据 tts_core.tts_all_in_one 的实际参数顺序调整
        res = tts_core.tts_all_in_one(
            text, speaker, emotion, 1.0, ref_audio, ref_text, model_size, safe_seed
        )
        return res[0], safe_seed # 返回音频路径和使用的种子
    except Exception as e:
        gr.Warning(f"💡 合成提示: {str(e)}")
        return None, safe_seed

# =========================================================
# UI 布局 (严格保留原有结构)
# =========================================================
with gr.Blocks(title="Qwen3 TTS Advanced") as demo:
    gr.Markdown("# 🎙️ Qwen3. NEURAL VOICE ENGINE")
    
    with gr.Row():
        # 左侧控制栏 (scale=1)
        with gr.Column(scale=1):
            mode_nav = gr.Radio(["官方角色", "语音设计", "零样本克隆"], label="功能导航", value="官方角色")
            model_size = gr.Radio(["Pro", "Lite"], label="模型规格", value="Lite")
            seed_input = gr.Number(value=-1, label="随机种子 (-1 为随机)", precision=0)
            
            # --- 分组: 官方角色 (增加语言联动) ---
            with gr.Group(visible=True) as group_custom:
                gr.Markdown("### 👤 官方角色设置")
                language_select = gr.Dropdown(LANGUAGE_CHOICES, value="Chinese", label="选择语言")
                speaker_select = gr.Dropdown(SPEAKER_MAP["Chinese"], value="Vivian", label="内置说话人")
                emotion_select = gr.Dropdown(EMOTION_EXAMPLES, value="Normal tone", label="语气控制")

            # --- 分组: 语音设计 ---
            with gr.Group(visible=False) as group_design:
                gr.Markdown("### 🎨 语音设计")
                design_instruct = gr.Textbox(label="描述声音特征", placeholder="例如：声音低沉的男性")

            # --- 分组: 零样本克隆 ---
            with gr.Group(visible=False) as group_clone:
                gr.Markdown("### 🧬 声音克隆")
                ref_audio = gr.Audio(label="上传参考音频 (3-10秒)", type="filepath")
                ref_text = gr.Textbox(label="参考文本", placeholder="识别中或手动输入...", interactive=True)

        # 右侧内容区 (scale=2)
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="输入待合成文本", lines=8, placeholder="输入您想让 AI 说的话...")
            gen_btn = gr.Button("🚀 开始生成", variant="primary")
            output_audio = gr.Audio(label="输出音频", interactive=False)
            used_seed_res = gr.Number(label="本次使用的种子", interactive=False)

    # =========================================================
    # 事件绑定
    # =========================================================
    # 1. 语言与角色联动
    language_select.change(fn=update_speakers, inputs=language_select, outputs=speaker_select)

    # 2. 模式切换显示
    mode_nav.change(fn=switch_ui_mode, inputs=mode_nav, outputs=[group_custom, group_design, group_clone])
    
    # 3. 参考音频识别
    ref_audio.change(fn=tts_core.transcribe_audio, inputs=ref_audio, outputs=ref_text)

    # 4. 生成按钮
    gen_btn.click(
        fn=validated_tts,
        inputs=[
            text_input, mode_nav, model_size, language_select, 
            speaker_select, emotion_select, design_instruct, 
            ref_audio, ref_text, seed_input
        ],
        outputs=[output_audio, used_seed_res]
    )

if __name__ == "__main__":
    demo.launch(server_port=9860, theme=gr.themes.Soft())