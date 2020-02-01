import os
import yaml
from pprint import pprint

CONFIG_NAME = "config.yaml"

class ConfigReader():
    def __init__(self, profile_name="DEFAULT"):
        self.config_data = None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(dir_path, CONFIG_NAME)
        with open(filepath, "r") as f:
            self.config_data = yaml.safe_load(f)["DEFAULT"]
        if profile_name != "DEFAULT":
            with open(filepath, "r") as f:
                new_dict = yaml.safe_load(f)[profile_name]
                self.update(new_dict)
    
    def update(self, new_dict):
        self.config_data = recursive_dict_update(self.config_data, new_dict)
    
def recursive_dict_update(base_dict, update_dict):
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            base_dict[k] = recursive_dict_update(base_dict.get(k, {}), v)
        else:
            base_dict[k] = v
    return base_dict

if __name__ == "__main__":
    config_data = ConfigReader().config_data
    pprint(config_data)