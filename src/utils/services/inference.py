# nim_client.py
import os
import yaml
import json
from typing import Dict
from openai import OpenAI
from pydantic import ValidationError

from src.root import PROJECT_ROOT
from src.utils import config
from src.utils.services.logger_config import logger
from src.utils.api.schemas import InferenceResponse


class NIMClient:
    """Client for interacting with the NVIDIA NIM Chat Completions API."""

    def __init__(self):
        self.api_key = config.NVIDIA_API_KEY
        self.base_url = config.NVIDIA_BASE_URL
        self.model = config.LLM_MODEL
        self.timeout = 120  # default request timeout
        # Initialize LLM client
        self.client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY,
        )

    def _nim_headers(self) -> Dict[str, str]:
        """Prepare and return request headers for NIM API calls."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def clean_json_fence(self, input_str: str) -> str:
        return input_str.strip().removeprefix("```json").removesuffix("```").strip()

    @staticmethod
    def load_yaml_config() -> dict:
        """Load YAML configuration file from src/prompts."""
        config_path = os.path.join(PROJECT_ROOT, "src", "prompts", "prompt.yaml")
        logger.info(f"Loading the prompts from yaml file: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def generate_prompt(self, history_text: str, question: str, context: str) -> str:
        """Load the prompt template and format it with user input."""
        system_prompt = self.load_yaml_config().get("system_prompt")
        user_prompt = self.load_yaml_config().get("user_prompt")
        user_prompt = user_prompt.format(
            history_text=history_text, question=question, context=context
        )

        logger.info("Generated the user prompt")

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def chat_completion(self, history_text: str, question: str, context: str) -> str:
        """
        Send a chat completion request to the NIM API.
        Mimics OpenAI-compatible POST /v1/chat/completions.
        """
        prompts = self.generate_prompt(
            history_text=history_text, question=question, context=context
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompts.get("system_prompt")},
                {"role": "user", "content": prompts.get("user_prompt")},
            ],
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_tokens=config.MAX_TOKENS,
            stream=False,
        )
        raw_output = completion.choices[0].message.content.strip()
        raw_output = self.clean_json_fence(input_str=raw_output)

        try:
            parsed_json = json.loads(raw_output)
            validated_output = InferenceResponse(**parsed_json)
            logger.info(f"Valid string is generated")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid model output: {e}")
            raise ValueError("LLM output validation failed.") from e

        return validated_output.response
