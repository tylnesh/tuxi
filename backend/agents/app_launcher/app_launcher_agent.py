import os
import subprocess
import configparser
from pathlib import Path
from rapidfuzz import fuzz, process
from collections import defaultdict

class AppLauncherAgent:
    def __init__(self, threshold=90):
        self.app_index = {}               # main app info: {canonical_name: exec_cmd}
        self.alias_to_name = {}           # alias index: {alias: canonical_name}
        self.threshold = threshold
        self._build_index()

    def _find_desktop_files(self):
        paths = ["/usr/share/applications"]
                #  , str(Path.home() / ".local/share/applications")]
        for path in paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith(".desktop"):
                        yield os.path.join(path, file)

    def _parse_desktop_file(self, filepath):
        config = configparser.ConfigParser(interpolation=None)
        try:
            config.read(filepath, encoding='utf-8')
            section = "Desktop Entry"
            name = config.get(section, "Name", fallback=None)
            exec_cmd = config.get(section, "Exec", fallback=None)
            if not name or not exec_cmd:
                return None

            # Clean Exec command
            exec_cmd = exec_cmd.split()[0]

            # Gather all useful aliases
            aliases = set()
            aliases.add(name)
            aliases.add(config.get(section, "GenericName", fallback=""))
            # aliases.add(config.get(section, "Comment", fallback=""))
            keywords = config.get(section, "Keywords", fallback="")
            if keywords:
                aliases.update(keywords.replace(";", " ").split())

            # Remove empty aliases
            aliases = {alias.strip() for alias in aliases if alias.strip()}

            return name, exec_cmd, aliases
        except Exception:
            return None

    def _build_index(self):
        for file in self._find_desktop_files():
            parsed = self._parse_desktop_file(file)
            if not parsed:
                continue
            name, exec_cmd, aliases = parsed
            self.app_index[name] = exec_cmd
            for alias in aliases:
                self.alias_to_name[alias.lower()] = name

    def launch(self, user_input: str):
        if not self.alias_to_name:
            print("No applications found.")
            return

        aliases = list(self.alias_to_name.keys())
        match, score, _ = process.extractOne(user_input, aliases, scorer=fuzz.ratio)
        if match and score >= self.threshold:
            app_name = self.alias_to_name[match]
            command = self.app_index.get(app_name)
            try:
                subprocess.Popen([command])
                print(f"Launched: {app_name} (matched: '{match}')")
            except Exception as e:
                print(f"Failed to launch {app_name}: {e}")
        else:
            print("No matching application found.")

# Example usage
if __name__ == "__main__":
    agent = AppLauncherAgent()
    while True:
        user_input = input("What app do you want to launch (or 'quit')? ")
        if user_input.lower() in {"quit", "exit"}:
            break
        agent.launch(user_input)
