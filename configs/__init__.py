from typing import Dict, Type
from .config_base import ConfigBase
from .config_supervised import ConfigSupervised

task_to_config: Dict[str, Type[ConfigBase]] = {
    'supervised': ConfigSupervised,
}
