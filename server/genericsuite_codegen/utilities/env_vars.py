import os
import json
from pathlib import Path
from typing import Any

local_envvars = None


def load_envvars():
    lib_root_dir = Path(Path(__file__).parent).parent
    with open(f"{lib_root_dir}/assets/main_config.json", "r") as f:
        return json.load(f)


def get_envvar(var_name: str, default: Any = None):
    global local_envvars
    if local_envvars is None:
        local_envvars = load_envvars()
    return local_envvars.get(var_name, os.getenv(var_name, default))
