from dotenv import load_dotenv

load_dotenv()

import os
import json
import yaml
import time
from openai import OpenAI
from pydantic import ValidationError

from src.utils.logger_config import LoggerConfig
from src.utils.models import SqlResponse, NaturalResponse
from src.config import PROJECT_ROOT


class Text2Sql:
    def __init__(self):
        # Initialize logger
        self.logger = LoggerConfig().logger

        # Initialize LLM client
        self.client = OpenAI(
            base_url=os.getenv("NVIDIA_LLM_ENDPOINT"),
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
        )

    @staticmethod
    def load_yaml_config(command_fname: str) -> dict:
        """Load a YAML configuration file from src/configs."""
        config_path = os.path.join(PROJECT_ROOT, "src", "configs", command_fname)
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    
    def clean_json_fence(self,input_str: str) -> str:
        return (
            input_str.strip()
            .removeprefix("```json").removesuffix("```")
            .strip()
        )

    def generate_prompt(self, user_query: str, schema_info: str) -> str:
        """Load the prompt template and format it with user input."""
        prompt_template = self.load_yaml_config("tsql_prompt.yaml").get("tsql_prompt")
        return prompt_template.format(user_query=user_query, schema_info=schema_info)

    def run(self, user_query: str, schema_info: str) -> str:
        """Run SQL inference and return validated SQL query."""
        prompt = self.generate_prompt(user_query, schema_info)

        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            # model="google/gemma-3-1b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.7,
            max_tokens=1024,
            stream=False,
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000  # Convert to milliseconds

        raw_output = completion.choices[0].message.content.strip()
        raw_output = self.clean_json_fence(input_str=raw_output)

        try:
            parsed_json = json.loads(raw_output)
            validated_output = SqlResponse(**parsed_json)
            self.logger.info(f"Valid SQL is generated")
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Invalid model output: {e}")
            raise ValueError("LLM output validation failed.") from e

        return validated_output.sql, latency_ms


class Sql2Text:
    def __init__(self):
        # Initialize logger
        self.logger = LoggerConfig().logger

        # Initialize LLM client
        self.client = OpenAI(
            base_url=os.getenv("NVIDIA_LLM_ENDPOINT"),
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
        )

    @staticmethod
    def load_yaml_config(command_fname: str) -> dict:
        """Load a YAML configuration file from src/configs."""
        config_path = os.path.join(PROJECT_ROOT, "src", "configs", command_fname)
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def generate_prompt(self, user_query: str, sql_query: str, sql_answer: str) -> str:
        """Load the prompt template and format it with user input."""
        prompt_template = self.load_yaml_config("sqlt_prompt.yaml").get("sqlt_prompt")
        return prompt_template.format(
            user_query=user_query, sql_query=sql_query, sql_answer=sql_answer
        )

    def run(self, user_query: str, sql_query: str, sql_answer: str) -> str:
        """Run SQL inference and return validated SQL query."""
        prompt = self.generate_prompt(
            user_query=user_query, sql_query=sql_query, sql_answer=sql_answer
        )

        completion = self.client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.7,
            max_tokens=1024,
            stream=False,
        )

        raw_output = completion.choices[0].message.content.strip()

        try:
            parsed_json = json.loads(raw_output)
            validated_output = NaturalResponse(**parsed_json)
            self.logger.info(f"Valid Natural response is generated")
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Invalid model output: {e}")
            raise ValueError("LLM output validation failed.") from e

        return validated_output.text
