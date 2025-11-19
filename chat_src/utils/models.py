from pydantic import BaseModel, Field


class RagCase(BaseModel):
    question: str


class RagEval(BaseModel):
    question: str
    db_schema: str
    db_name: str
    model: str
    shot: int


class DbSummarize(BaseModel):
    db_name: str


class SqlResponse(BaseModel):
    sql: str


class NaturalResponse(BaseModel):
    text: str


class DeleteCaseRequest(BaseModel):
    collection_names: str = Field(
        ...,
        description="Comma-separated list of collections to delete from: e.g., 'evidence,narratives,case_chat'",
    )
    db_name: str = Field(..., description="The case_id to delete documents for")


class NimEvalType(BaseModel):
    complexity: str = Field(
        ..., description="The complexity of the test dataset being used"
    )
    shot: int = Field(..., description="The number of examples being sent to the LLM")
    model: str = Field(..., description="The name of the LLM")
