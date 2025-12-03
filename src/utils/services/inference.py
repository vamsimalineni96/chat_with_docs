# nim_client.py
import os
import yaml
import json
from typing import Dict
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from pydantic import ValidationError
from src.root import PROJECT_ROOT
from src.utils.errors import InferenceError
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
            base_url=self.base_url,
            api_key=self.api_key,
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

    def generate_prompt(self, history_text: str, question: str, context: str) -> dict:
        """Load the prompt template and format it with user input."""
        cfg = self.load_yaml_config()
        system_prompt = cfg.get("system_prompt")
        user_prompt = cfg.get("user_prompt")

        if not system_prompt or not user_prompt:
            logger.error("Prompt YAML missing 'system_prompt' or 'user_prompt'.")
            raise ValueError("Prompt configuration is incomplete.")

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

        # --- Prompt generation (YAML / template issues) ---
        try:
            prompts = self.generate_prompt(
                history_text=history_text, question=question, context=context
            )
        except FileNotFoundError as e:
            logger.error("Prompt YAML file not found: %s", e)
            raise InferenceError("Prompt configuration file not found.") from e
        except yaml.YAMLError as e:
            logger.error("Failed to parse YAML prompt file: %s", e)
            raise InferenceError("Prompt configuration is invalid YAML.") from e
        except Exception as e:
            logger.exception("Unexpected error while generating prompts: %s", e)
            raise InferenceError("Unexpected error while generating prompts.") from e

        try:
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
        except AuthenticationError as e:
            logger.error("Authentication with NIM/OpenAI endpoint failed: %s", e)
            raise InferenceError("Authentication with NIM endpoint failed.") from e
        except BadRequestError as e:
            logger.error("Bad request to NIM chat completions: %s", e)
            logger.debug("Question: %r", question[:200])
            raise InferenceError("Bad request to NIM chat completions.") from e
        except RateLimitError as e:
            logger.warning("Rate limit hit on NIM chat completions: %s", e)
            # Let caller decide any backoff/retry policy
            raise InferenceError("Rate limit hit on NIM chat completions.") from e
        except APIConnectionError as e:
            logger.error("Connection error calling NIM chat completions: %s", e)
            raise InferenceError("Connection error calling NIM chat completions.") from e
        except APIError as e:
            status = getattr(e, "status_code", None)
            logger.error("APIError from NIM chat completions (status=%s): %s", status, e)
            raise InferenceError("Upstream NIM API error.") from e
        except Exception as e:
            logger.exception("Unexpected error during NIM chat completion: %s", e)
            raise InferenceError("Unexpected error during NIM chat completion.") from e

        # --- Output cleaning + JSON + Pydantic validation ---
        raw_output = self.clean_json_fence(input_str=raw_output)

        try:
            parsed_json = json.loads(raw_output)
            validated_output = InferenceResponse(**parsed_json)
            logger.info("Valid model output generated and validated.")
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error("Invalid model output: %s", e)
            logger.debug("Raw output snippet: %r", raw_output[:500])
            raise InferenceError("LLM output validation failed.") from e
        except Exception as e:
            logger.exception("Unexpected error while validating model output: %s", e)
            raise InferenceError("Unexpected error while validating model output.") from e

        return validated_output.response
