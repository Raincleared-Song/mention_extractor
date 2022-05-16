from typing import Dict, Type
from .config_base import ConfigBase
from .config_supervised import ConfigSupervised
from .config_fewshot import ConfigFewshot

task_to_config: Dict[str, Type[ConfigBase]] = {
    'supervised': ConfigSupervised,
    'fewshot': ConfigFewshot,
}
