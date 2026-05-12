import os
import yaml
from typing import Dict

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.root import PROJECT_ROOT
from src.utils.errors import InferenceError
from src.utils import config
from src.utils.services.logger_config import logger


class NIMClient:
    """LangChain-backed client for NVIDIA NIM chat completions.

    A single LCEL chain: prompt | ChatNVIDIA | StrOutputParser. Returns plain
    prose grounded only in retrieved context — no JSON envelope, no separate
    query-expansion call.
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

        self._prompt_cfg = self._load_yaml_config()
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._prompt_cfg["system_prompt"]),
                ("human", self._prompt_cfg["user_prompt"]),
            ]
        )
        self._chain = self._prompt | self.llm | StrOutputParser()

    @staticmethod
    def _load_yaml_config() -> Dict:
        config_path = os.path.join(PROJECT_ROOT, "src", "prompts", "prompt.yaml")
        logger.info("Loading prompts from yaml file: %s", config_path)
        try:
            with open(config_path, "r") as fh:
                cfg = yaml.safe_load(fh)
        except FileNotFoundError as e:
            logger.error("Prompt YAML file not found: %s", e)
            raise InferenceError("Prompt configuration file not found.") from e
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML prompt file: %s", e)
            raise InferenceError("Prompt configuration is invalid YAML.") from e

        if not cfg or not cfg.get("system_prompt") or not cfg.get("user_prompt"):
            logger.error("Prompt YAML missing 'system_prompt' or 'user_prompt'.")
            raise InferenceError("Prompt configuration is incomplete.")
        return cfg

    def chat_completion(self, history_text: str, question: str, context: str) -> str:
        """Invoke the answer chain and return plain prose."""
        try:
            answer = self._chain.invoke(
                {
                    "history_text": history_text,
                    "question": question,
                    "context": context,
                }
            )
        except Exception as e:
            logger.exception("Error during NIM chat completion via LangChain: %s", e)
            raise InferenceError("NIM chat completion failed.") from e

        return (answer or "").strip()
