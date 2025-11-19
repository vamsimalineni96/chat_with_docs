from dotenv import load_dotenv

load_dotenv()

import os
import json
import yaml
import time
from openai import OpenAI
from pydantic import ValidationError

from prune_src.utils.logger_config import LoggerConfig
from prune_src.utils.models import SchemaPruneResponse
from prune_src.config import PROJECT_ROOT


class SchemaPrune:
    def __init__(self, model):
        # Initialize logger
        self.logger = LoggerConfig().logger
        self.model = model

        # Initialize LLM client
        self.client = OpenAI(
            base_url=os.getenv("NVIDIA_LLM_ENDPOINT"),
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
        )

    @staticmethod
    def load_yaml_config(command_fname: str) -> dict:
        """Load a YAML configuration file from src/configs."""
        config_path = os.path.join(PROJECT_ROOT, "prune_src", "configs", command_fname)
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def clean_json_fence(self, input_str: str) -> str:
        return input_str.strip().removeprefix("```json").removesuffix("```").strip()

    def generate_prompt(self, user_query: str, schema_info: str) -> str:
        """Load the prompt template and format it with user input."""
        prompt_template = self.load_yaml_config(f"schema_prune_prompt.yaml").get(
            "schema_pruning_prompt"
        )
        return prompt_template.format(user_query=user_query, schema_info=schema_info)

    def run(self, user_query: str, schema_info: str) -> str:
        """Run SQL inference and return validated SQL query."""
        self.logger.info("Generating prompt")
        prompt = self.generate_prompt(user_query, schema_info)
        self.logger.info(f"Running inference using: {self.model.lower().strip()}")

        if self.model.lower().strip() == "llama":
            llm_name = os.getenv("llama")
        elif self.model.lower().strip() == "gemma":
            llm_name = os.getenv("gemma")
        elif self.model.lower().strip() == "qwen":
            llm_name = os.getenv("qwen")

        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=llm_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.1,
            max_tokens=1024,
            stream=False,
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000  # Convert to milliseconds

        raw_output = completion.choices[0].message.content.strip()
        raw_output = self.clean_json_fence(input_str=raw_output)

        try:
            parsed_json = json.loads(raw_output)
            validated_output = SchemaPruneResponse(**parsed_json)
            self.logger.info(f"Valid response is generated")
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Invalid model output: {e}")
            raise ValueError("LLM output validation failed.") from e

        return validated_output.pruned_schema, latency_ms
