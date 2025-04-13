import os
import yaml


class ConfigManager:
    DEFAULT_CONFIG_YAML = """
api:
  host: localhost
  port: 11434
audio:
  sample_rate: 16000
  channels: 1
  buffer_size: 1024
  frame_duration_ms: 30
  pause_timeout: 1.0
  language: auto
  confidence_threshold: 0.5
  min_words: 3
  output_dir: transcripts
  vad:
    aggressiveness: 2
  model:
    name: medium
nlp:
  model: gemma3:27b
"""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = None
        self.DEFAULT_CONFIG = yaml.safe_load(self.DEFAULT_CONFIG_YAML)  # Parse YAML into a dictionary
        self.load_config()

    def load_config(self):
        """Load the configuration from the file or create a default one."""
        if not os.path.exists(self.config_file):
            # Create the directory for the config file if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            # Write the default configuration to the file
            with open(self.config_file, "w") as f:
                yaml.dump(self.DEFAULT_CONFIG, f)
            print(f"Default configuration created at {self.config_file}")
            self.config = self.DEFAULT_CONFIG
        else:
            try:
                with open(self.config_file, "r") as f:
                    self.config = yaml.safe_load(f) or self.DEFAULT_CONFIG
            except yaml.YAMLError as e:
                print(f"Error loading configuration file: {e}")
                self.config = self.DEFAULT_CONFIG

    def get(self, key: str, default=None):
        """Get a configuration value."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k, default)
        return value

    def save(self):
        """Save the current configuration to the file."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(self.config, f)
        except Exception as e:
            print(f"Error saving configuration: {e}")