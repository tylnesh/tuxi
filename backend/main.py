import os
import yaml
import threading
import time
import queue
from rapidfuzz import fuzz  # For fuzzy matching
from config_manager import ConfigManager

# Import audio modules
from audio.audio_capture import AudioCapture
from audio.vad import VoiceActivityDetector
from audio.file_handler import FileHandler
from audio.transcriber import TranscriptionEngine

# Import NLP modules
from nlp.nlp_processor import NLPProcessor
from nlp.intent_parser import IntentParser

# Initialize the configuration manager
CONFIG_FILE = "./config/config.yaml"
config_manager = ConfigManager(CONFIG_FILE)

# Access configuration values
SAMPLE_RATE = config_manager.get("audio.sample_rate", 16000)
CHANNELS = config_manager.get("audio.channels", 1)
BUFFER_SIZE = config_manager.get("audio.buffer_size", 1024)
FRAME_DURATION_MS = config_manager.get("audio.frame_duration_ms", 30)
PAUSE_TIMEOUT = config_manager.get("audio.pause_timeout", 1.0)
LANGUAGE = config_manager.get("audio.language", "auto")
CONFIDENCE_THRESHOLD = config_manager.get("audio.confidence_threshold", 0.5)
MIN_WORDS = config_manager.get("audio.min_words", 3)
OUTPUT_DIR = config_manager.get("audio.output_dir", "transcripts")
MODEL_NAME = config_manager.get("audio.model.name", "medium")
VAD_AGGRESSIVENESS = config_manager.get("audio.vad.aggressiveness", 2)

# Initialize NLP components
nlp_processor = NLPProcessor(config_file=CONFIG_FILE)
intent_parser = IntentParser()

# Global listening state: "idle", "active", or "processing"
listening_state = "idle"
WAKE_WORD_TARGET = "hey tuxi"
WAKE_FUZZY_THRESHOLD = 50  # For wake word detection

stop_prompt = False  # Global flag to stop prompt processing

def is_wake_word_detected(transcript: str, target: str = WAKE_WORD_TARGET, threshold: int = WAKE_FUZZY_THRESHOLD) -> bool:
    """
    Uses fuzzy matching on the first two words of the transcript.
    """
    words = transcript.lower().split()
    if len(words) < 2:
        return False
    transcript_start = " ".join(words[:2])
    score = fuzz.ratio(transcript_start, target)
    return score >= threshold

def process_prompt(command: str):
    """
    Processes the given command in a separate thread.
    Streams the NLP response in real-time and outputs the final intent.
    Resets the listening state to idle when done.
    """
    global listening_state, stop_prompt
    stop_prompt = False  # Reset the stop flag at the start of processing
    full_response_chunks = []

    def stream_callback(chunk: str):
        global stop_prompt
        if stop_prompt:
            raise StopIteration  # Stop streaming if the stop flag is set
        print(chunk, end="", flush=True)
        full_response_chunks.append(chunk)

    try:
        full_response = nlp_processor.query_stream(command, stream_callback)
        if not full_response:
            print("\n[Error] No response received from the model.")
            listening_state = "idle"
            return
        print("\n[Final LLM Response]:", full_response)
        intent, details = intent_parser.parse_intent(full_response)
        print("[Detected Intent]:", intent)
        print("[Intent Details]:", details)
        print("-" * 40)
    except StopIteration:
        print("\n[Info] Prompt processing stopped.")
    finally:
        listening_state = "idle"

def process_command(transcript: str):
    """
    Called with each transcribed snippet.
    Uses fuzzy matching to detect the wake word and processes commands accordingly.
    """
    global listening_state, stop_prompt
    transcript = transcript.strip()
    print("\n[Transcribed]:", transcript)

    # Check if the transcript starts with the wake word.
    if is_wake_word_detected(transcript):
        words = transcript.split()
        # Remove the first two words (assumed wake word) and clean punctuation.
        command = " ".join(words[2:]).strip(" ,.!?")
        if command:
            # Check if the immediate command equals "stop" exactly.
            if "stop" in command.lower():
                print("[Cancellation] 'Stop' command received via wake word.")
                listening_state = "idle"
                nlp_processor.query_stop()
                stop_prompt = True
            else:
                print("[Activation] Wake word detected with immediate command:", command)
                if listening_state != "processing":
                    listening_state = "processing"
                    threading.Thread(target=process_prompt, args=(command,), daemon=True).start()
                else:
                    print("[Info] Already processing a command.")
        else:
            print("[Activation] Wake word detected. Awaiting command...")
            listening_state = "active"
        return

    # If already active and additional speech is received, process it as a command.
    if listening_state == "active":
        if transcript:
            print("[Processing] Command received in active mode:", transcript)
            listening_state = "processing"
            threading.Thread(target=process_prompt, args=(transcript,), daemon=True).start()
        return

    # If idle and no wake word detected, ignore the input.
    if listening_state == "idle":
        print("[Info] Not activated. Say 'Hey Tuxi' to activate.")

# Create a shared queue for audio chunks.
audio_queue = queue.Queue()

# Initialize audio components.
audio_capture = AudioCapture(SAMPLE_RATE, CHANNELS, BUFFER_SIZE, audio_queue)
vad_detector = VoiceActivityDetector(
    aggressiveness=VAD_AGGRESSIVENESS,
    sample_rate=SAMPLE_RATE,
    frame_duration_ms=FRAME_DURATION_MS
)
file_handler = FileHandler(OUTPUT_DIR, CHANNELS, SAMPLE_RATE)

# Initialize the transcription engine with our NLP callback.
transcriber = TranscriptionEngine(
    audio_queue,
    vad_detector,
    file_handler,
    SAMPLE_RATE,
    int(SAMPLE_RATE * FRAME_DURATION_MS / 1000),
    PAUSE_TIMEOUT,
    LANGUAGE,
    CONFIDENCE_THRESHOLD,
    MIN_WORDS,
    model_name=MODEL_NAME,
    callback=process_command
)

# Start the transcription engine in a separate thread.
transcription_thread = threading.Thread(target=transcriber.process_audio, daemon=True)
transcription_thread.start()

# Start capturing audio.
audio_capture.start_stream()

print("🎙️ Listening with WebRTC VAD... Speak and pause to transcribe. Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    audio_capture.stop_stream()
    print("\n🛑 Stopped.")
