from dotenv import load_dotenv

load_dotenv()

import os
import asyncio
import litellm

from chat_src.db_crew.db_agent import DbCrew
from chat_src.utils.logger_config import LoggerConfig
from chat_src.utils.sql_handler import DatabaseHandler
from chat_src.utils.vectorstore import get_vectorstore_handler


# LiteLLM configuration for NVIDIA
litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")
litellm.set_verbose = False

# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


async def summarize_schema(db_schema):
    tsql_input = {"db_schema": db_schema}
    tsql_result = await DbCrew().crew().kickoff_async(inputs=tsql_input)

    if hasattr(tsql_result, "pydantic") and tsql_result.pydantic:
        logger.info(f"Pydantic output type: {type(tsql_result.pydantic)}")
        summary = tsql_result.pydantic.Summary
    else:
        summary = "The pydantic has failed"

    return summary


async def run_summarizer(db_name):
    db_handler = DatabaseHandler(db_name)
    db_schema = db_handler.get_db_schema_json()

    result = await summarize_schema(db_schema)

    vector_store = get_vectorstore_handler()
    vector_store.add_summary(db_name=db_name, text=result)

    
