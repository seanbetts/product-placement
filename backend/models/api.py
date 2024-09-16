from pydantic import BaseModel

class NameUpdate(BaseModel):
    name: str