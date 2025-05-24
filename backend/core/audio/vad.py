import webrtcvad

class VoiceActivityDetector:
    def __init__(self, aggressiveness, sample_rate, frame_duration_ms):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

    def is_speech(self, frame_bytes):
        return self.vad.is_speech(frame_bytes, self.sample_rate)
