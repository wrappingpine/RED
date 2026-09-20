"""Air Mouse Profile Manager

Handles creation, editing, deletion, import and export of named profiles.
"""

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

import tomllib

from .config import Config, ConfigManager

logger = logging.getLogger(__name__)


class ProfileInfo:
    """Metadata for a profile."""
    name: str
    source_config_path: Optional[str] = None  # None for default; otherwise absolute path
    created_at: str = ""
    description: str = ""

    def __init__(self, name: str, description: str = "", source_config_path: Optional[str] = None):
        self.name = name
        self.description = description
        self.source_config_path = source_config_path
        import time
        self.created_at = str(int(time.time()))


class ProfileManager:
    """Manages named config profiles with import/export/delete/create lifecycle."""

    def __init__(self, config_manager: Any = None):
        if config_manager is None:
            from ..config import ConfigManager
            config_manager = ConfigManager()
        self.config_manager = config_manager

    def get_profiles_dir(self) -> Path:
        """Return path to profiles directory."""
        return self.config_manager.profiles_dir

    def list_profiles(self) -> List[ProfileInfo]:
        """List all saved profiles."""
        profiles_dir = self.config_manager.profiles_dir
        if not profiles_dir.exists():
            return []
        profiles = []
        for f in profiles_dir.iterdir():
            if f.is_file() and f.suffix == ".toml" and not f.name.startswith("exported_"):
                name_stem = f.stem
                profiles.append(ProfileInfo(name=name_stem, source_config_path=str(f)))
        return profiles

    def create(self, name: str, description: str = "") -> Optional[Path]:
        """Create a new profile named <name> with current config values."""
        profiles_dir = self.config_manager.profiles_dir
        profiles_dir.mkdir(parents=True, exist_ok=True)

        filename = profiles_dir / f"{name}.toml"
        if filename.exists():
            logger.warning(f"Profile '{name}' already exists.")
            return None

        profile_info = ProfileInfo(name=name, description=description)
        # Store profile info inline as TOML metadata header
        cfg = self.config_manager.load()
        data = self.config_manager.to_dict(cfg)

        # Write profile as TOML with simple key-value structure
        with open(filename, "w") as f:
            f.write(f'# Air Mouse Profile: {name}\n')
            f.write(f'# Description: {description}\n')
            f.write(f'# Created: {profile_info.created_at}\n')
            # Embed config data as nested table
            for section, section_data in data.items():
                f.write(f'\n[{section}]\n')
                for key, val in section_data.items():
                    f.write(f'{key} = {self._value_to_toml(val)}\n')

        return filename

    @staticmethod
    def _value_to_toml(val: Any) -> str:
        """Convert a Python value to a TOML-compatible string."""
        import numbers
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, numbers.Integral):
            return str(int(val))
        if isinstance(val, numbers.Real):
            return str(float(val))
        return f'"{str(val)}"'

    def delete(self, name: str) -> bool:
        """Delete a profile by name."""
        profiles_dir = self.config_manager.profiles_dir
        filename = profiles_dir / f"{name}.toml"
        if not filename.exists():
            logger.warning(f"Profile '{name}' not found.")
            return False
        try:
            filename.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete profile '{name}': {e}")
            return False

    def import_profile(self, source_path: Union[str, Path]) -> Optional[Path]:
        """Import a profile from an external TOML path."""
        source = Path(source_path)
        if not source.exists():
            logger.error(f"Source profile path does not exist: {source_path}")
            return None

        profiles_dir = self.config_manager.profiles_dir
        profiles_dir.mkdir(parents=True, exist_ok=True)

        filename = profiles_dir / source.name
        try:
            shutil.copy2(str(source), str(filename))
            return filename
        except Exception as e:
            logger.error(f"Failed to import profile from {source_path}: {e}")
            return None

    def export(self, name: str, dest_path: Union[str, Path]) -> bool:
        """Export a profile to a destination path."""
        profiles_dir = self.config_manager.profiles_dir
        filename = profiles_dir / f"{name}.toml"
        dest = Path(dest_path)

        if not filename.exists():
            logger.warning(f"Profile '{name}' not found.")
            return False

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(filename), str(dest))
            return True
        except Exception as e:
            logger.error(f"Failed to export profile '{name}': {e}")
            return False

    def load_profile(self, name: str) -> Optional[Config]:
        """Load a profile and return a Config object."""
        profiles_dir = self.config_manager.profiles_dir
        filename = profiles_dir / f"{name}.toml"

        if not filename.exists():
            logger.warning(f"Profile '{name}' not found.")
            return None

        try:
            with open(filename, "rb") as f:
                data = f.read()
            # Strip metadata headers
            text = data.decode("utf-8")
            # Remove header lines
            lines = text.split("\n")
            data_lines = []
            in_config = False
            current_section = None
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# Air Mouse Profile:"):
                    continue  # skip headers
                if stripped.startswith("# Created:"):
                    continue  # skip headers
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_config = True
                    current_section = stripped
                    data_lines.append(line)
                    continue
                if in_config:
                    if stripped and not stripped.startswith("#"):
                        data_lines.append(line)
                    elif not stripped:
                        in_config = False
            # Parse remaining data
            import tomllib
            raw = "\n".join(data_lines)
            parsed = tomllib.loads(raw) if sys.version_info >= (3, 11) else tomli.loads(raw)
            return self.config_manager.from_dict(parsed)
        except Exception as e:
            logger.error(f"Failed to load profile '{name}': {e}")
            return None

    def get_active_config(self) -> Config:
        """Return the currently active (loaded or default) config merged with profile data.

        Profiles can selectively override individual sections.
        """
        cfg = self.config_manager.load()
        # List profiles; this could be extended to merge active profile settings
        # into the active config
        return cfg