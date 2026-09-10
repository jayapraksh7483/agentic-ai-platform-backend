"""
Manager Agent package (Phase 4).

Public surface used by the rest of the app:
    manager.service.orchestrate(db, user_input, ...)                  -> OrchestrateResponse
    manager.service.get_execution(db, execution_id)                   -> ExecutionStatusResponse | None
    manager.service.approve_pending_agent(db, execution_id, approved) -> OrchestrateResponse
"""
from . import service  # noqa: F401