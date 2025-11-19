from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

import litellm
import os
from crewai.flow import Flow, start, listen

from chat_src.utils.inference import Text2Sql, Sql2Text
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


class RagFlow(Flow):

    @start()
    def fetch_lead(self):
        user_query = self.state.get("user_query", "default_question")
        logger.info(f"Fetched lead: question={user_query}")
        return {"question": user_query}

    @listen(fetch_lead)
    async def get_db_id(self):
        vector_store = get_vectorstore_handler()
        user_query = self.state["user_query"]
        logger.info(f"Fetching the db id given the user query: {user_query}")
        db_name = vector_store.query(query=user_query)
        self.state["db_name"] = db_name
        return {"db_name": db_name}

    @listen(get_db_id)
    async def get_db_schema(self):
        db_name = self.state["db_name"]
        logger.info(f"Accessing the schema for the database: {db_name}")
        sql_handler = DatabaseHandler(db_name)
        logger.info("Retreiving schema for the given database")
        self.state["schema_info"] = sql_handler.get_db_schema_json()

    @listen(get_db_schema)
    async def generate_sql_query(self):

        user_query = self.state["user_query"]
        schema_info = self.state["schema_info"]

        tsql = Text2Sql()

        tsql_output = tsql.run(user_query=user_query, schema_info=schema_info)
        logger.info(f"the tsql output is : {tsql_output}")

        self.state["sql_query"] = tsql_output

        return tsql_output

    @listen(generate_sql_query)
    async def execute_sql_query(self):
        db_name = self.state["db_name"]
        sql_handler = DatabaseHandler(db_name)
        logger.info("Executing the generated sql query")
        self.state["sql_answer"] = sql_handler.execute_command(
            query=self.state["sql_query"]
        )
        return {"message": "Generated sql query is now executed"}

    @listen(execute_sql_query)
    async def answer_query(self):
        sql_answer = self.state.get("sql_answer", "")
        sql_query = self.state.get("sql_query")
        user_query = self.state.get("user_query")

        sqlt = Sql2Text()

        sqlt_output = sqlt.run(
            user_query=user_query, sql_answer=sql_answer, sql_query=sql_query
        )

        return sqlt_output
