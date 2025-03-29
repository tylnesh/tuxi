import os
import wave
import numpy as np

class FileHandler:
    def __init__(self, output_dir, channels, sample_rate):
        self.output_dir = output_dir
        self.channels = channels
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

    def save_wav(self, audio_np, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # assuming 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())

    def save_txt(self, text, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
