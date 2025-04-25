# audio/transcriber.py
import time
import numpy as np
from datetime import datetime
import queue
import whisper
import threading

CONTROL_COMMANDS = ["stop", "hey tuxi", "tuxi"]

class TranscriptionEngine:
    def __init__(self, audio_queue, vad_detector, file_handler,
                 sample_rate, frame_size,
                 pause_timeout, language, confidence_threshold, min_words,
                 model_name="medium", callback=None):
        self.audio_queue = audio_queue
        self.vad_detector = vad_detector
        self.file_handler = file_handler
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.pause_timeout = pause_timeout
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.min_words = min_words
        self.callback = callback  # Store the callback
        self.buffer = []
        self.last_voice_time = None
        self.last_state = None
        self.whisper_model = whisper.load_model(model_name).to("cuda")
        self.stop_event = threading.Event()  # New: Event to signal stopping

    def stop(self):
        """Signal the thread to stop."""
        self.stop_event.set()

    def process_audio(self):
        while not self.stop_event.is_set():  # Check if stop signal is set
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                chunk = None

            if chunk is not None:
                chunk = chunk.flatten().astype(np.float32)
                frame = (chunk[:self.frame_size] * 32767).astype(np.int16).tobytes()
                if len(frame) < self.frame_size * 2:
                    continue

                if self.vad_detector.is_speech(frame):
                    self.buffer.append(chunk)
                    self.last_voice_time = time.time()
                    if self.last_state != "speech":
                        print("🟢 Speaking")
                        self.last_state = "speech"
                else:
                    if self.last_state != "silence":
                        print("⚫ Silence")
                        self.last_state = "silence"

                if self.buffer and self.last_voice_time and time.time() - self.last_voice_time > self.pause_timeout:
                    print("⏸️ Transcribing after pause...")
                    full_audio = np.concatenate(self.buffer).astype(np.float32)
                    self.buffer.clear()
                    self.last_voice_time = None

                    result = self.whisper_model.transcribe(
                        full_audio, 
                        language=None if self.language == "auto" else self.language,
                        task="translate",
                    )
                    text = result["text"].strip()
                    segments = result.get("segments", [])
                    avg_conf = (np.mean([seg.get("confidence", 1.0) for seg in segments])
                                if segments else 1.0)

                    if len(text.split()) < self.min_words and not (
                        text.lower().strip() in CONTROL_COMMANDS or text.lower().startswith("hey tuxi")
                    ):
                        print(f"[Skipped: too short] {text}\n")
                        continue
                    if avg_conf < self.confidence_threshold:
                        print(f"[Skipped: low confidence {avg_conf:.2f}] {text}\n")
                        continue

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"{self.file_handler.output_dir}/segment_{timestamp}"
                    self.file_handler.save_wav(full_audio, f"{filename_base}.wav")
                    self.file_handler.save_txt(text, f"{filename_base}.txt")

                    print(f"[Saved] {filename_base}.wav + .txt")
                    print(f"📜 Transcription: {text} (conf: {avg_conf:.2f})\n")

                    # Call the callback with the transcribed text if provided
                    if self.callback:
                        self.callback(text)

        print("🛑 Stopping transcription engine...")
