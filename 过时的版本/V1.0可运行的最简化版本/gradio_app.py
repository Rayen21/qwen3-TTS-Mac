import gradio as gr
from tts_core import tts

with gr.Blocks(title="Qwen3 TTS (Apple Silicon)") as demo:
    gr.Markdown("## 🎙️ Qwen3 TTS（Apple Silicon / MLX）")

    with gr.Row():
        text_input = gr.Textbox(
            label="输入文本",
            lines=5,
            placeholder="请输入要合成语音的文本"
        )

    generate_btn = gr.Button("生成语音")

    audio_output = gr.Audio(
        label="生成结果",
        type="filepath",
        autoplay=False
    )

    generate_btn.click(
        fn=tts,
        inputs=text_input,
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=9860,
        share=False
    )
