from dotenv import load_dotenv
load_dotenv()

import os
from crewai import LLM

class AzureOpenAIConfig:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        # Set up the LLM for Azure OpenAI GPT-4o
        self.llm = LLM(
            model=f"azure/{self.deployment_name}",
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version
        )
    def _print_config(self):
        print(self.api_key, self.api_base, self.api_version, self.deployment_name)
# Usage
azure_openai_config = AzureOpenAIConfig()
azure_llm= azure_openai_config.llm