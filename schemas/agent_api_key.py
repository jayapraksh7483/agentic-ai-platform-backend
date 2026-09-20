from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AgentAPIKeyCreateRequest(BaseModel):
    """
    expires_at is optional. When omitted, the key has no automatic
    expiry and remains valid until revoked.
    """

    expires_at: Optional[datetime] = None


class AgentAPIKeyOut(BaseModel):
    id: str
    agent_id: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AgentAPIKeyCreatedResponse(AgentAPIKeyOut):
    """
    api_key is returned only once, immediately after creation.
    It is never stored in plaintext and cannot be retrieved later.
    """

    api_key: str
    invoke_path: str
