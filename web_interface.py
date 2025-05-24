# gradio_chat_app.py (Corrected for Gradio 4.x)

import gradio as gr
import whisper
from queue import Queue
from backend.text_chat import TextChat
from backend.voice_chat import VoiceChat

# --- Output buffer to collect streamed outputs ---
class OutputBuffer:
    def __init__(self):
        self.queue = Queue()
        self.buffer = []

    def push(self, text):
        self.queue.put(text)
        self.buffer.append(text)

    def stream(self):
        while not self.queue.empty():
            yield self.queue.get()

    def get_history(self):
        return self.buffer

# --- Initialize components ---
text_chat = TextChat()
voice_chat = VoiceChat()
output_buffer = OutputBuffer()

# --- Monkey patch for streaming process_prompt ---
def patched_process_prompt(self, prompt, output_buffer: OutputBuffer):
    for chunk in self._generate_response_chunks(prompt):
        output_buffer.push(chunk)
        yield chunk

TextChat.process_prompt_streaming = patched_process_prompt
VoiceChat.process_prompt_streaming = patched_process_prompt

# --- Handlers for Gradio ---
def handle_text_input(text, history):
    output_buffer.queue.queue.clear()
    output_buffer.buffer.clear()

    response_stream = text_chat.process_prompt_streaming(text, output_buffer)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": ""})

    def response_gen():
        result = ""
        for chunk in response_stream:
            result += chunk
            history[-1]["content"] = result
            yield history

    return response_gen, history

def handle_audio_input(audio_path, history):
    if audio_path is None:
        return "No audio received.", history

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()

    if not transcript:
        return "Could not transcribe.", history

    voice_chat.process_command(transcript)
    history.append({"role": "user", "content": f"[Voice]: {transcript}"})
    history.append({"role": "assistant", "content": "[processing...]"})
    return f"Transcribed: {transcript}", history

# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Chat with Tuxi (Text + Voice)")

    with gr.Tabs():
        with gr.Tab("Text Chat"):
            chat_history = gr.State([])
            chatbot = gr.Chatbot(label="Tuxi Chat", type="messages")
            text_input = gr.Textbox(placeholder="Type your message...", label="You")

            text_input.submit(handle_text_input, [text_input, chat_history], [chatbot, chat_history])

        with gr.Tab("Voice Chat"):
            mic_input = gr.Audio(label="Speak", format="wav", streaming=False)
            voice_output = gr.Textbox(label="Transcription")
            mic_input.change(handle_audio_input, [mic_input, chat_history], [voice_output, chat_history])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
