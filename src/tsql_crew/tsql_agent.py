import os
from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from src.utils.logger_config import LoggerConfig

from src.utils.model_selector import get_llm_model

# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class SqlOutput(BaseModel):
    sql: str = Field(
        ...,
        description="Sql query generated from the natural language prompt",
    )

def nl_to_sql_task_callback(output):
    logger.info("nl_to_sql_task is complete")


@CrewBase
class TsqlCrew:
    """Crew for generating reports and logging transcripts into vector DB"""

    @agent
    def NL2SQL_Agent(self) -> Agent:
        logger.info("Initializing NL2SQL_Agent...")
        agent = Agent(
            config=self.agents_config["NL2SQL_Agent"],
            verbose=True,
            llm=get_llm_model(),
        )
        logger.info("NL2SQL_Agent initialized successfully.")
        return agent
    @task
    def nl_to_sql_task(self) -> Task:
        logger.info("Creating nl to sql task...")
        task = Task(
            config=self.tasks_config["nl_to_sql_task"],
            output_pydantic=SqlOutput,
            callback=nl_to_sql_task_callback,
        )
        logger.info("nl to sql task created successfully.")
        return task
    

    ##################################################################
    @crew
    def crew(self) -> Crew:
        """Creates the RAG crew for vector DB operations"""
        logger.info("Creating RagCrew with agents and tasks...")
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
        logger.info("RagCrew initialized successfully.")
        return crew
