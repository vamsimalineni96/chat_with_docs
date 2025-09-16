from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

import litellm
import os
from crewai.flow import Flow, start, listen

from src.tsql_crew.tsql_agent import TsqlCrew

from src.utils.logger_config import LoggerConfig
from src.utils.sql_handler import SqlHandler

# LiteLLM configuration for NVIDIA
litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")
litellm.set_verbose = False


# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class RagFlow(Flow):

    @start()
    def fetch_lead(self):
        """Fetches the lead data (case_id and question)."""
        case_id = self.state.get("case_id", "default_case")
        question = self.state.get("user_query", "default_question")
        logger.info(f"Fetched lead: case_id={case_id}, question={question}")
        return {"case_id": case_id, "question": question}

    @listen(fetch_lead)
    async def get_db_schema(self):
        sql_handler = SqlHandler()
        logger.info("Retreiving schema for the given database")
        self.state["schema_info"] = sql_handler.get_db_schema_json()

    @listen(fetch_lead)
    async def generate_sql_query(self):
        tsql_input = {
            "user_query": self.state["user_query"],
            "schema_info": self.state["schema_info"],
        }
        tsql_result = await TsqlCrew().crew().kickoff_async(inputs=tsql_input)

        if hasattr(tsql_result, "pydantic") and tsql_result.pydantic:
            logger.info(f"Pydantic output type: {type(tsql_result.pydantic)}")
            tsql_output = tsql_result.pydantic.sql
        else:
            tsql_output = "The pydantic has failed"

        self.state["sql_query"] = tsql_output

        return tsql_output

    @listen(generate_sql_query)
    async def execute_sql_query(self):
        sql_handler = SqlHandler()
        logger.info("Executing the generated sql query")
        return {
            "generated_query": self.state["sql_query"],
            "executed": sql_handler.execute_command(query=self.state["sql_query"]),
        }
