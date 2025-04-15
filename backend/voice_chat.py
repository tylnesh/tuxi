import threading
from backend.base_chat import BaseChat
from backend.config_manager import ConfigManager
from backend.audio.audio_capture import AudioCapture
from backend.audio.vad import VoiceActivityDetector
from backend.audio.file_handler import FileHandler
from backend.audio.transcriber import TranscriptionEngine
from rapidfuzz import fuzz
import queue
import time

# TODO: Replace the wakeword detection with something more light-weight so it can run constantly. 
# Maybe https://github.com/dscripka/openWakeWord 

class VoiceChat(BaseChat):
    listening_state = "idle"
    WAKE_WORD_TARGET = "hey tuxi"
    WAKE_FUZZY_THRESHOLD = 50

    def __init__(self, config_file="./backend/config/config.yaml"):
        super().__init__(config_file)
        self.stop_prompt = False

        config = ConfigManager(config_file)

        sample_rate = config.get("audio.sample_rate", 16000)
        channels = config.get("audio.channels", 1)
        buffer_size = config.get("audio.buffer_size", 1024)
        frame_duration_ms = config.get("audio.frame_duration_ms", 30)
        pause_timeout = config.get("audio.pause_timeout", 1.0)
        language = config.get("audio.language", "auto")
        confidence_threshold = config.get("audio.confidence_threshold", 0.5)
        min_words = config.get("audio.min_words", 3)
        output_dir = config.get("audio.output_dir", "transcripts")
        model_name = config.get("audio.model.name", "medium")
        vad_aggressiveness = config.get("audio.vad.aggressiveness", 2)

        audio_queue = queue.Queue()
        self.audio_capture = AudioCapture(sample_rate, channels, buffer_size, audio_queue)
        vad = VoiceActivityDetector(vad_aggressiveness, sample_rate, frame_duration_ms)
        file_handler = FileHandler(output_dir, channels, sample_rate)

        self.transcriber = TranscriptionEngine(
            audio_queue,
            vad,
            file_handler,
            sample_rate,
            int(sample_rate * frame_duration_ms / 1000),
            pause_timeout,
            language,
            confidence_threshold,
            min_words,
            model_name,
            callback=self.process_command
        )

    def is_wake_word_detected(self, transcript):
        words = transcript.lower().split()
        if len(words) < 2:
            return False
        return fuzz.ratio(" ".join(words[:2]), self.WAKE_WORD_TARGET) >= self.WAKE_FUZZY_THRESHOLD

    def process_command(self, transcript):
        transcript = transcript.strip()
        print("\n[Transcribed]:", transcript)

        if self.is_wake_word_detected(transcript):
            words = transcript.split()
            command = " ".join(words[2:]).strip(" ,.!?")

            if command:
                if "stop" in command.lower():
                    print("[Cancellation] 'Stop' command received.")
                    self.listening_state = "idle"
                    self.nlp_processor.query_stop()
                    self.stop_prompt = True
                else:
                    print("[Activation] Wake word detected with command:", command)
                    if self.listening_state != "processing":
                        self.listening_state = "processing"
                        threading.Thread(target=self.process_prompt, args=(command,), daemon=True).start()
                    else:
                        print("[Info] Already processing a command.")
            else:
                print("[Activation] Wake word detected. Awaiting command...")
                self.listening_state = "active"
            return

        if self.listening_state == "active" and transcript:
            print("[Processing] Command received in active mode:", transcript)
            self.listening_state = "processing"
            threading.Thread(target=self.process_prompt, args=(transcript,), daemon=True).start()
            return

        if self.listening_state == "idle":
            print("[Info] Not activated. Say 'Hey Tuxi' to activate.")

    def start(self):
        threading.Thread(target=self.transcriber.process_audio, daemon=True).start()
        self.audio_capture.start_stream()
        print("🎙️ Listening with WebRTC VAD... Speak and pause to transcribe. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.audio_capture.stop_stream()
            print("\n🛑 Stopped.")


if __name__ == "__main__":
    VoiceChat().start()
