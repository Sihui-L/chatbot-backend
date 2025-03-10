from typing import Literal

from pydantic import UUID4, BaseModel


class Request(BaseModel):
    request_id: UUID4


class SyncResponse(BaseModel):
    request_id: UUID4
    data: str


class AsyncResponse(BaseModel):
    request_id: UUID4


class SocketPayload1(BaseModel):
    event_type: Literal["init_conneciton"]
    content: str


class SocketPayload2(BaseModel):
    event_type: Literal["action"]
    data: str
