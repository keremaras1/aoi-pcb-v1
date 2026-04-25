"""Hierarchical JSON configuration loader with dot-notation access."""

import json
from typing import Any


class Config:
    """Load and provide dot-notation access to a hierarchical JSON config file.

    Nested dictionaries in the JSON are recursively converted to Config instances,
    allowing attribute-style traversal (e.g. ``config.training.n_epochs``).

    Example::

        config = Config("config.json")
        lr = config.training.optimizer_lr
        kwargs = config.get_init_kwargs("generator.train_data")
    """

    def __init__(self, config_path: str = "config.json") -> None:
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file.
        """
        with open(config_path, "r") as f:
            config_data = json.load(f)
        self._set_attributes(config_data)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Config has no attribute '{name}'")

    def _set_attributes(self, config_data: dict[str, Any]) -> None:
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Recursively create Config instances for nested dictionaries
                setattr(self, key, Config.from_dict(value))
            else:
                setattr(self, key, value)

    def get_init_kwargs(self, key: str) -> dict[str, Any]:
        """Return a nested config section as a plain dict for ``**kwargs`` unpacking.

        Supports dot-separated keys for nested traversal.

        Args:
            key: Dot-separated path to a nested section, e.g. ``"generator.train_data"``.

        Returns:
            Dictionary of key-value pairs from the target section.

        Raises:
            ValueError: If any part of the path does not exist, or if the final
                value is not a nested section (i.e. not a Config instance).
        """
        keys = key.split(".")
        nested_section = self

        for k in keys:
            if not hasattr(nested_section, k):
                raise ValueError(f"Key '{key}' not found in the configuration.")
            nested_section = getattr(nested_section, k)

        if not isinstance(nested_section, Config):
            raise ValueError(f"Key '{key}' must point to a nested dictionary.")

        return nested_section.__dict__

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> "Config":
        """Create a Config instance directly from a dictionary.

        Args:
            data_dict: Dictionary to convert into a Config instance.

        Returns:
            A new Config instance with attributes set from the dictionary.
        """
        config_instance = cls.__new__(cls)  # Bypass __init__ to avoid re-opening a file
        config_instance._set_attributes(data_dict)
        return config_instance
