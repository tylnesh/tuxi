import threading
import time
import queue
from audio.audio_capture import AudioCapture
from audio.vad import VoiceActivityDetector
from audio.file_handler import FileHandler
from audio.transcriber import TranscriptionEngine

# CONFIG
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SIZE = 1024
FRAME_DURATION_MS = 30
PAUSE_TIMEOUT = 1.0
LANGUAGE = "auto"
CONFIDENCE_THRESHOLD = 0.5
MIN_WORDS = 3
OUTPUT_DIR = "transcripts"
MODEL_NAME = "medium"

# Shared queue for audio chunks
audio_queue = queue.Queue()

# Initialize components
audio_capture = AudioCapture(SAMPLE_RATE, CHANNELS, BUFFER_SIZE, audio_queue)
vad_detector = VoiceActivityDetector(aggressiveness=2, sample_rate=SAMPLE_RATE, frame_duration_ms=FRAME_DURATION_MS)
file_handler = FileHandler(OUTPUT_DIR, CHANNELS, SAMPLE_RATE)
transcriber = TranscriptionEngine(audio_queue, vad_detector, file_handler,
                                  SAMPLE_RATE, int(SAMPLE_RATE * FRAME_DURATION_MS / 1000),
                                  PAUSE_TIMEOUT, LANGUAGE, CONFIDENCE_THRESHOLD, MIN_WORDS,
                                  model_name=MODEL_NAME)

# Start transcription in a separate thread
transcription_thread = threading.Thread(target=transcriber.process_audio, daemon=True)
transcription_thread.start()

# Start audio capture stream
audio_capture.start_stream()

print("🎙️ Listening with WebRTC VAD... Speak and pause to transcribe. Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    audio_capture.stop_stream()
    print("\n🛑 Stopped.")
