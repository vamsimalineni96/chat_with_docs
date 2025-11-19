from pydantic import BaseModel

class SchemaPruneResponse(BaseModel):
    pruned_schema: str

