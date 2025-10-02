from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

import litellm
import os
from crewai.flow import Flow, start, listen

from src.utils.inference import Text2Sql
from src.utils.logger_config import LoggerConfig
from src.utils.sql_handler import DatabaseHandler

# LiteLLM configuration for NVIDIA
litellm.api_key = os.getenv("NVIDIA_NIM_API_KEY")
litellm.api_base = os.getenv("NVIDIA_LLM_ENDPOINT")
litellm.set_verbose = False


# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class EvalFlow(Flow):

    @start()
    def fetch_lead(self):
        user_query = self.state.get("user_query", "default_question")
        db_schema = self.state.get("db_schema", "default_schema")
        db_name = self.state.get("db_name", "default_dbname")
        logger.info(
            f"Fetched lead: question={user_query}, db_schema: {db_schema}, db_name: {db_name} "
        )
        return {"question": user_query}

    @listen(fetch_lead)
    async def generate_sql_query(self):

        user_query = self.state["user_query"]
        db_schema = self.state["db_schema"]

        tsql = Text2Sql()

        tsql_output = tsql.run(user_query=user_query, schema_info=db_schema)
        logger.info(f"the tsql output is : {tsql_output}")

        self.state["sql_query"] = tsql_output

        return tsql_output

    @listen(generate_sql_query)
    async def execute_sql_query(self):
        db_name = self.state["db_name"]
        sql_handler = DatabaseHandler(db_name, test=True)
        logger.info("Executing the generated sql query")
        self.state["sql_answer"] = sql_handler.execute_command(
            query=self.state["sql_query"]
        )
        return {
            "sql_query": self.state["sql_query"],
            "sql_answer": self.state["sql_answer"],
        }
