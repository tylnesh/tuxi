import sounddevice as sd
import numpy as np
import queue

class AudioCapture:
    def __init__(self, samplerate, channels, buffer_size, audio_queue):
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_size = buffer_size
        self.audio_queue = audio_queue

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start_stream(self):
        self.stream = sd.InputStream(callback=self.audio_callback,
                                       channels=self.channels,
                                       samplerate=self.samplerate,
                                       blocksize=self.buffer_size)
        self.stream.start()

    def stop_stream(self):
        self.stream.stop()
