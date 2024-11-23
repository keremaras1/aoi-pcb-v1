import json


class Config:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        self._set_attributes(config_data)

    def _set_attributes(self, config_data):
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Recursively create Config instances for nested dictionaries
                setattr(self, key, Config.from_dict(value))
            else:
                setattr(self, key, value)

    def get_init_kwargs(self, key):
        if not hasattr(self, key):
            raise ValueError(f"Key '{key}' not found in the configuration.")

        nested_section = getattr(self, key)
        if not isinstance(nested_section, Config):
            raise ValueError(f"Key '{key}' must point to a nested dictionary.")

        return nested_section.__dict__

    @classmethod
    def from_dict(cls, data_dict):
        config_instance = cls.__new__(cls)  # Avoid calling __init__
        config_instance._set_attributes(data_dict)
        return config_instance
