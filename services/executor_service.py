"""
Agent Execution Engine.

Standard contract:
  1. Find the requested agent from the Registry
  2. Validate the agent is ACTIVE
  3. Validate the execution request
  4. Send the request to the runtime (Gemini)
  5. Receive + process the response
  6. Handle errors / timeouts
  7. Store the execution record
  8. Return the final result
"""

import time
import json

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.config import settings

from models.agent import Agent, AgentStatus
from models.execution import AgentExecution, ExecutionStatus

from schemas.agent import (
    AgentExecuteRequest,
    AgentExecuteResponse,
)

from agent_runtime.runtime import agent_runtime


def _extract_text(user_input) -> str:
    if isinstance(user_input, str):
        return user_input

    if isinstance(user_input, dict):
        if "query" in user_input:
            return str(user_input["query"])

        if "text" in user_input:
            return str(user_input["text"])

        return json.dumps(user_input)

    return str(user_input)


class AgentNotFoundError(Exception):
    pass


class AgentInactiveError(Exception):
    pass


def _call_llm(
    db: Session,
    agent: Agent,
    system_prompt: str,
    user_input: str,
) -> str:
    """
    Execute the agent through the LangGraph runtime.
    """

    result = agent_runtime.execute(
        agent_id=str(agent.id),
        user_input=user_input,
        provider=agent.provider,
        model=agent.model,
        system_prompt=system_prompt,
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    return str(result.get("output", ""))


def execute_agent(
    db: Session,
    agent_id: str,
    request: AgentExecuteRequest,
    user_id: Optional[int] = None,
) -> AgentExecuteResponse:
    """
    Execute an agent.

    When user_id is supplied, the agent must belong to that user.

    User-facing API endpoints must always provide the authenticated
    user's ID.
    """

    # ========================================================
    # 1. Find the agent
    # ========================================================

    query = db.query(Agent).filter(
        Agent.id == agent_id
    )

    # ========================================================
    # 2. Enforce ownership
    # ========================================================

    if user_id is not None:
        query = query.filter(
            Agent.created_by == user_id
        )

    agent: Optional[Agent] = query.first()

    if not agent:
        raise AgentNotFoundError(
            f"Agent '{agent_id}' not found"
        )

    # ========================================================
    # 3. Validate active
    # ========================================================

    if agent.status != AgentStatus.ACTIVE:
        raise AgentInactiveError(
            f"Agent '{agent_id}' is not active"
        )

    # ========================================================
    # 4. Validate request
    # ========================================================

    input_text = _extract_text(request.input)

    if not input_text or not input_text.strip():
        raise ValueError(
            "Execution input cannot be empty"
        )

    # ========================================================
    # 5. Create execution record
    # ========================================================

    execution = AgentExecution(
        agent_id=agent.id,
        user_id=user_id,
        input_payload=(
            json.dumps(request.input)
            if isinstance(request.input, dict)
            else request.input
        ),
        status=ExecutionStatus.PENDING,
    )

    db.add(execution)
    db.commit()
    db.refresh(execution)

    start = time.monotonic()

    try:

        # ====================================================
        # 6. Execute through runtime
        # ====================================================

        output_text = _run_with_timeout(
            lambda: _call_llm(
                db,
                agent,
                agent.system_prompt,
                input_text,
            ),
            timeout_seconds=settings.EXECUTION_TIMEOUT_SECONDS,
        )

        # ====================================================
        # 7. Successful execution
        # ====================================================

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        output_obj = {
            "result": output_text
        }

        execution.output_payload = json.dumps(
            output_obj
        )

        execution.status = ExecutionStatus.SUCCESS
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            output=output_obj,
            latency_ms=latency_ms,
        )

    # ========================================================
    # 8. Timeout
    # ========================================================

    except TimeoutError as e:

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        execution.status = ExecutionStatus.TIMEOUT
        execution.error_message = str(e)
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            error=str(e),
            latency_ms=latency_ms,
        )

    # ========================================================
    # 9. General execution failure
    # ========================================================

    except Exception as e:

        latency_ms = (
            time.monotonic() - start
        ) * 1000

        execution.status = ExecutionStatus.FAILED
        execution.error_message = str(e)
        execution.end_time = datetime.now(timezone.utc)
        execution.latency_ms = latency_ms

        db.commit()

        return AgentExecuteResponse(
            execution_id=execution.id,
            agent_id=agent.id,
            status=execution.status.value,
            error=str(e),
            latency_ms=latency_ms,
        )


def _run_with_timeout(
    fn,
    timeout_seconds: int,
):
    """
    Runs fn() in a worker thread and raises TimeoutError
    if it exceeds timeout_seconds.
    """

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    ) as ex:

        future = ex.submit(fn)

        try:
            return future.result(
                timeout=timeout_seconds
            )

        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Agent execution exceeded "
                f"{timeout_seconds}s timeout"
            )