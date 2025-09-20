import os
from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from src.utils.logger_config import LoggerConfig

from src.utils.model_selector import get_llm_model

# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class DbOutput(BaseModel):
    Domain: str = Field(
        ..., description="A short phrase describing the database domain"
    )
    Summary: str = Field(
        ..., description="One or two detailed paragraphs about the database"
    )

def db_task_callback(output):
    logger.info("db_task is complete")


@CrewBase
class DbCrew:
    """Crew for generating reports and logging transcripts into vector DB"""

    @agent
    def Db_Agent(self) -> Agent:
        logger.info("Initializing Db_Agent...")
        agent = Agent(
            config=self.agents_config["Db_Agent"],
            verbose=True,
            llm=get_llm_model(),
        )
        logger.info("Db_Agent initialized successfully.")
        return agent
    @task
    def db_task(self) -> Task:
        logger.info("Creating nl to sql task...")
        task = Task(
            config=self.tasks_config["db_task"],
            output_pydantic=DbOutput,
            callback=db_task_callback,
        )
        logger.info("db task created successfully.")
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
        logger.info("DbCrew initialized successfully.")
        return crew
