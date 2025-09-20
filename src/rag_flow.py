from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

import litellm
import os
from openai import OpenAI
from crewai.flow import Flow, start, listen

from src.tsql_crew.tsql_agent import TsqlCrew

from src.utils.logger_config import LoggerConfig
from src.utils.sql_handler import DatabaseHandler
from src.utils.vectorstore import get_vectorstore_handler

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
        self.state["db_name"] =db_name
        return {"db_name": db_name}

    @listen(get_db_id)
    async def get_db_schema(self):
        db_name=self.state["db_name"]
        logger.info(f"Accessing the schema for the database: {db_name}")
        sql_handler = DatabaseHandler(db_name)
        logger.info("Retreiving schema for the given database")
        self.state["schema_info"] = sql_handler.get_db_schema_json()

    @listen(get_db_schema)
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
        db_name=self.state["db_name"]
        sql_handler = DatabaseHandler(db_name)
        logger.info("Executing the generated sql query")
        self.state["sql_answer"]=sql_handler.execute_command(query=self.state["sql_query"])
        return {"message":"Generated sql query is now executed"}

    @listen(execute_sql_query)
    async def answer_query(self):
        sql_answer=self.state.get("sql_answer","")
        sql_query=self.state.get("sql_query")
        user_query= self.state.get("user_query")
        
        client = OpenAI(
            base_url=os.getenv("NVIDIA_LLM_ENDPOINT"),
            api_key=os.getenv("NVIDIA_NIM_API_KEY"),
        )

        prompt = f""" 
            You are an expert data analyst who converts database results into clear, natural language answers.

            ### Task:
            You will be given:
            1. A user's natural language question.
            2. The SQL query that was generated to answer the question.
            3. The result of executing the SQL query.

            Your job is to carefully read the SQL output and provide a **concise, natural language answer** to the user's question.  
            - Do NOT show the SQL query or output in the final response.  
            - If the result is empty or does not provide enough information, say that clearly.  
            - Make the answer sound like a direct, friendly explanation.

            ### Input:
            User Question: {user_query}
            Generated SQL Query: {sql_query}
            SQL Output: {sql_answer}

            ### Output:
            A natural language answer to the user question based ONLY on the SQL output.
            """

        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False,
        )

        return completion.choices[0].message.content