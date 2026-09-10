from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from models.execution import ExecutionStatus


class ExecutionOut(BaseModel):
    id: str
    agent_id: str
    user_id: Optional[int]
    input_payload: str
    output_payload: Optional[str]
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime]
    latency_ms: Optional[float]
    error_message: Optional[str]

    model_config = ConfigDict(from_attributes=True)
