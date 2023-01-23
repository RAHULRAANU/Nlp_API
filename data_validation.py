# for data handling and validation

from pydantic import BaseModel, constr
from typing import List


class Textrequest(BaseModel): # NOQA : E302 for hide warning
    # sentence: str
    sentence: list[str]


# class Textrequest(BaseModel):  # NOQA : E302 for hide warning
#     sentence: constr(min_length = 10)


class decode_text(BaseModel):   # NOQA : E302
    cybersecurity_text: str
    Not_cyber_security: str

class Output(BaseModel):
    extractions: List[decode_text]
