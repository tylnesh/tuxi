import os
import yaml

class ConfigManager:
    DEFAULT_CONFIG = {
        "api": {
            "host": "localhost",
            "port": 11434
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "buffer_size": 1024,
            "frame_duration_ms": 30,
            "pause_timeout": 1.0,
            "language": "auto",
            "confidence_threshold": 0.5,
            "min_words": 3,
            "output_dir": "transcripts",
            "vad": {
                "aggressiveness": 2
            },
            "model": {
                "name": "medium"
            }
        },
        "nlp": {
            "model": "gemma3:27b"
        }
    }

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = None
        self.load_config()

    def load_config(self):
        """Load the configuration from the file or create a default one."""
        if not os.path.exists(self.config_file):
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, "w") as f:
                yaml.dump(self.DEFAULT_CONFIG, f)
            print(f"Default configuration created at {self.config_file}")
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Get a configuration value."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value else default

    def save(self):
        """Save the current configuration to the file."""
        with open(self.config_file, "w") as f:
            yaml.dump(self.config, f)