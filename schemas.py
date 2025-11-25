from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    k: int = 5     
