from pydantic import BaseModel

class RagCase(BaseModel):
    question: str

