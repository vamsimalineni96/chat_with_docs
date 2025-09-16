import os
import yaml
from src.utils.azure_utils import azure_llm  # Assuming GPT-4 access
from src.utils.on_prem_model import on_prem_llm

def load_yaml_config() -> dict:
    # Get the root directory of the project (two levels up from utils)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Construct the path to configs/chose_model.yaml
    config_path = os.path.join(project_root, "configs", "chose_model.yaml")

    # Load the YAML
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_llm_model():
    config = load_yaml_config()
    print("Loaded YAML config:", config)  # 🔍 Debug print
    
    if config.get("gpt4", {}).get("choice", "").lower() == "yes":
        return azure_llm
    
    elif config.get("llama", {}).get("choice", "").lower() == "yes":
        llama_model = os.getenv("LLM_MODEL")
        if not llama_model:
            raise EnvironmentError("LLM_MODEL environment variable not set for llama.")
        return llama_model
    
    elif config.get("on_prem",{}).get("choice", "").lower() == "yes":
        return on_prem_llm
    else:
        raise ValueError("No valid model selected. Set 'choice: \"yes\"' for either gpt4 or llama in chose_model.yaml.")
