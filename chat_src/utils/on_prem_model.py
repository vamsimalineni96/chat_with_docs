from dotenv import load_dotenv
load_dotenv()

import os
from crewai import LLM

on_prem_endpoint=os.getenv("on_prem_endpoint")
on_prem_model=os.getenv("on_prem_model")

# Create your custom LLM
on_prem_llm= LLM(
    model=on_prem_model,
    api_key="dummy",
    base_url=on_prem_endpoint,
    temperature= 0.6,
    top_p=0.95,
)