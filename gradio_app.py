# gradio_app.py
import gradio as gr
import tts_core
import random

# 配置数据
SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"],
}
LANGUAGE_CHOICES = list(SPEAKER_MAP.keys())
EMOTIONS = ["Normal tone", "Sad", "Excited", "Angry", "Whispering"]

def update_speakers(lang):
    """语言联动"""
    speakers = SPEAKER_MAP.get(lang, [])
    return gr.update(choices=speakers, value=speakers[0] if speakers else None)

def switch_ui_mode(mode_label):
    """模式切换并锁定 Pro 后端"""
    mapping = {"官方角色": "Pro-Custom", "语音设计": "Pro-Design", "零样本克隆": "Pro-Clone"}
    tts_core.LOCAL_MODEL_NAME = mapping[mode_label]
    return [gr.update(visible=(mode_label == "官方角色")),
            gr.update(visible=(mode_label == "语音设计")),
            gr.update(visible=(mode_label == "零样本克隆"))]

def validated_tts(text, mode, lang, speaker, emotion, design_text, ref_audio, ref_text, seed):
    """封装调用后端"""
    if not text or text.strip() == "":
        gr.Warning("⚠️ 文本为空")
        return None, -1
    
    # 语言/角色/情感逻辑处理
    instruct = design_text if mode == "语音设计" else emotion
    
    # 调用后端 Pro 逻辑
    audio_path, used_seed = tts_core.tts_all_in_one(
        text, speaker, instruct, 1.0, ref_audio, ref_text, seed
    )
    return audio_path, used_seed

with gr.Blocks(title="Qwen3 Pro TTS") as demo:
    gr.Markdown("# 🎙️ Qwen3. NEURAL VOICE ENGINE (Pro 1.7B)")
    
    with gr.Row():
        # 左侧控制 (scale=1)
        with gr.Column(scale=1):
            mode_nav = gr.Radio(["官方角色", "语音设计", "零样本克隆"], label="功能导航", value="官方角色")
            seed_input = gr.Number(value=-1, label="随机种子 (-1 为随机)", precision=0)
            
            with gr.Group(visible=True) as group_custom:
                gr.Markdown("### 👤 角色设置")
                lang_sel = gr.Dropdown(LANGUAGE_CHOICES, value="Chinese", label="语言")
                spk_sel = gr.Dropdown(SPEAKER_MAP["Chinese"], value="Vivian", label="角色")
                emo_sel = gr.Dropdown(EMOTIONS, value="Normal tone", label="情感")

            with gr.Group(visible=False) as group_design:
                gr.Markdown("### 🎨 语音设计")
                design_input = gr.Textbox(label="声音描述", placeholder="例如：磁性男声")

            with gr.Group(visible=False) as group_clone:
                gr.Markdown("### 🧬 声音克隆")
                ref_aud = gr.Audio(label="参考音频", type="filepath")
                ref_txt = gr.Textbox(label="参考文本", interactive=True)

        # 右侧内容 (scale=2)
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="合成文本", lines=8, placeholder="输入内容...")
            gen_btn = gr.Button("🚀 开始生成 (Pro 1.7B)", variant="primary")
            out_aud = gr.Audio(label="输出音频", interactive=False)
            res_seed = gr.Number(label="所用种子", interactive=False)
            
    # --- 底部作者信息 ---
    gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; border-top: 1px solid #e5e5e5;">
            <p style="color: #666;">
                Developed by <b>Rayen21</b> | 
                <a href="https://github.com/Rayen21/qwen3-TTS-Mac" target="_blank" style="color: #4A90E2; text-decoration: none;">GitHub Project</a>
            </p>
        </div>
    """)        

    # 事件绑定
    lang_sel.change(fn=update_speakers, inputs=lang_sel, outputs=spk_sel)
    mode_nav.change(fn=switch_ui_mode, inputs=mode_nav, outputs=[group_custom, group_design, group_clone])
    ref_aud.change(fn=tts_core.transcribe_audio, inputs=ref_aud, outputs=ref_txt)
    
    gen_btn.click(
        fn=validated_tts,
        inputs=[text_input, mode_nav, lang_sel, spk_sel, emo_sel, design_input, ref_aud, ref_txt, seed_input],
        outputs=[out_aud, res_seed]
    )

if __name__ == "__main__":
    demo.launch(server_port=9860, theme=gr.themes.Soft())