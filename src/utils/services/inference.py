import os
import yaml
from typing import Dict

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.root import PROJECT_ROOT
from src.utils.errors import InferenceError
from src.utils import config
from src.utils.services.logger_config import logger


class NIMClient:
    """Thin wrapper around NVIDIA NIM (Llama 3.3 70B) for agentic chat.

    Exposes:
      - `self.llm`: the raw `ChatNVIDIA` instance. Callers bind tools via
        `nim_client.llm.bind_tools([...])` then `.invoke(messages)`.
      - `self.system_prompt`: the system prompt loaded from `prompt.yaml`.
    """

    def __init__(self):
        self.api_key = config.NVIDIA_API_KEY
        self.model = config.LLM_MODEL

        self.llm = ChatNVIDIA(
            model=self.model,
            api_key=self.api_key,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
        )
        self.system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        config_path = os.path.join(PROJECT_ROOT, "src", "prompts", "prompt.yaml")
        logger.info("Loading prompts from yaml file: %s", config_path)
        try:
            with open(config_path, "r") as fh:
                cfg: Dict = yaml.safe_load(fh) or {}
        except FileNotFoundError as e:
            logger.error("Prompt YAML file not found: %s", e)
            raise InferenceError("Prompt configuration file not found.") from e
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML prompt file: %s", e)
            raise InferenceError("Prompt configuration is invalid YAML.") from e

        system_prompt = cfg.get("system_prompt")
        if not system_prompt:
            logger.error("Prompt YAML missing 'system_prompt'.")
            raise InferenceError("Prompt configuration is incomplete.")
        return system_prompt
