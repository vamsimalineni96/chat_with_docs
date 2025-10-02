from pydantic import BaseModel, Field


class RagCase(BaseModel):
    question: str

class RagEval(BaseModel):
    question: str
    db_schema: str
    db_name :str

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
