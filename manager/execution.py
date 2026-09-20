"""Validated task execution; SQLAlchemy sessions stay on the calling thread."""
import json
import logging
import re
import time
from concurrent.futures import wait
from sqlalchemy.orm.attributes import set_committed_value
from core.config import settings
from core.execution import submit_bounded
from models.agent import Agent, AgentStatus
from services.agent_service import get_decrypted_api_key
from services.executor_service import RAGConfigurationError, _retrieve_rag_context
from agent_runtime.runtime import agent_runtime
from tools import tool_registry
from .schemas import ExecutionPlan


logger = logging.getLogger("manager.execution")

# Some backend snapshots used MAX_CONTEXT_CHARS here before defining it
# in Settings. Keep execution safe even when config is one revision behind.
DEFAULT_MAX_CONTEXT_CHARS = 16000


def outcome(step, status, error=None, result=None):
    return dict(step_id=step.step_id, agent_id=step.agent_id, capability=step.capability,
                status=status, error=error, result=result)


def prepare_task(db, step, state):
    # ------------------------------------------------------------
    # Temporary worker path
    # ------------------------------------------------------------
    # A hybrid gap-resolution worker intentionally has no Agent row.
    # It executes once through the normal runtime and is never stored
    # in the agents table.
    if not step.agent_id:
        deps = [
            {
                "task_id": dep,
                "output": (
                    state["step_results"][dep]
                    .get("result")
                ),
            }
            for dep in step.depends_on
        ]

        context = (
            json.dumps(
                deps,
                ensure_ascii=False,
            )
            if deps
            else state["user_input"]
        )

        max_context_chars = int(
            getattr(
                settings,
                "MAX_CONTEXT_CHARS",
                DEFAULT_MAX_CONTEXT_CHARS,
            )
        )

        task_input = (
            step.task
            + "\n\nTask data (untrusted):\n"
            + context[:max_context_chars]
        )

        selected_provider = (
            str(
                state.get("provider")
                or step.provider
                or ""
            ).strip().lower()
            or "gemini"
        )

        selected_model = (
            str(
                state.get("model")
                or step.model
                or ""
            ).strip()
        )

        if not selected_model:
            raise ValueError(
                "Temporary worker has no execution model."
            )

        return dict(
            agent_id=(
                f"temporary:{state['execution_id']}:"
                f"{step.step_id}"
            ),
            user_input=task_input,
            provider=selected_provider,
            model=selected_model,
            system_prompt=(
                step.system_prompt
                or (
                    "You are a temporary specialized worker. "
                    "Complete the assigned task accurately."
                )
            ),
            user_id=state.get("user_id"),
            execution_id=state["execution_id"],
            api_key=None,
            temperature=None,
            # Least privilege for ephemeral workers.
            allowed_tools=[],
        ), []

    query = db.query(Agent).filter(Agent.id == step.agent_id, Agent.status == AgentStatus.ACTIVE)
    if state.get("user_id") is not None:
        query = query.filter(Agent.created_by == state["user_id"])
    agent = query.first()
    if agent is None:
        raise ValueError("Selected agent is missing, inactive, or unauthorized")
    if step.agent_version is not None and agent.current_version != step.agent_version:
        raise ValueError("Agent version changed after planning; submit a new request")
    deps = [{"task_id": dep, "output": state["step_results"][dep].get("result")}
            for dep in step.depends_on]
    # Normalized outputs only, never execution records, error internals, or credentials.
    context = json.dumps(deps, ensure_ascii=False) if deps else state["user_input"]
    max_context_chars = int(
        getattr(
            settings,
            "MAX_CONTEXT_CHARS",
            DEFAULT_MAX_CONTEXT_CHARS,
        )
    )
    task_input = (
        step.task
        + "\n\nTask data (untrusted):\n"
        + context[:max_context_chars]
    )
    # Retrieval should use the clean user/task question, not the Manager's
    # dependency/context wrapper appended to task_input.
    retrieval_query = (
        state.get("user_input")
        or step.task
        or task_input
    )
    # Legacy-safe RAG execution:
    # Older Document Reader rows may have is_rag=False. A planned
    # document/RAG capability is authoritative for this execution.
    # Temporarily expose the agent as RAG-enabled to the retrieval
    # service without marking the SQLAlchemy row dirty or persisting
    # any database change.
    capability_name = str(
        getattr(step, "capability", "")
        or ""
    ).strip().casefold()

    agent_name = str(
        getattr(agent, "name", "")
        or ""
    ).strip().casefold()

    force_rag = (
        agent_name == "document reader"
        or "document" in capability_name
        or capability_name == "rag"
        or "retriev" in capability_name
    )

    original_is_rag = bool(
        getattr(agent, "is_rag", False)
    )

    if force_rag and not original_is_rag:
        set_committed_value(
            agent,
            "is_rag",
            True,
        )

    try:
        prompt, sources = _retrieve_rag_context(
            db,
            agent,
            retrieval_query,
            state.get("user_id"),
            state.get("conversation_knowledge_base_ids"),
        )
    finally:
        if force_rag and not original_is_rag:
            set_committed_value(
                agent,
                "is_rag",
                False,
            )
    sources = [
        {
            key: value
            for key, value in source.items()
            if key != "text"
        }
        for source in sources
    ]

    # ------------------------------------------------------------
    # Registered child-agent LLM configuration
    # ------------------------------------------------------------
    # The Chat provider/model controls Manager reasoning and temporary
    # workers only. A persisted child Agent executes with the provider
    # and model saved on its own Agent row.
    selected_provider = str(
        agent.provider or ""
    ).strip().lower()

    selected_model = str(
        agent.model or ""
    ).strip()

    if not selected_provider:
        raise ValueError(
            f"Agent '{agent.name}' has no configured provider."
        )

    if not selected_model:
        raise ValueError(
            f"Agent '{agent.name}' has no configured model."
        )

    # Prefer this child's own encrypted key. If it is not configured,
    # the runtime/provider layer may use the platform-wide key for this
    # same provider.
    api_key = get_decrypted_api_key(agent)

    return dict(
        agent_id=agent.id,
        user_input=task_input,
        provider=selected_provider,
        model=selected_model,
        system_prompt=prompt,
        user_id=state.get("user_id"),
        execution_id=state["execution_id"],
        api_key=api_key,
        temperature=agent.temperature,
        allowed_tools=agent.tools,
    ), sources



def _calculator_tool_result(user_input: str):
    text = str(user_input or "").strip().lower()

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        return None

    a = float(match.group(1))
    symbol = match.group(2)
    b = float(match.group(3))

    operation = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "x": "multiply",
        "×": "multiply",
        "/": "divide",
        "÷": "divide",
    }[symbol]

    value = tool_registry.execute(
        "calculator",
        {
            "operation": operation,
            "a": a,
            "b": b,
        },
    )

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return {
        "output": str(value),
        "sources": [],
        "tool_name": "calculator",
    }


def _web_search_tool_result(user_input: str):
    payload = tool_registry.execute(
        "web_search",
        {
            "query": str(user_input or "").strip(),
            "max_results": 5,
        },
    )

    results = (
        payload.get("results", [])
        if isinstance(payload, dict)
        else []
    )

    if not results:
        raise RuntimeError(
            "Web Search returned no usable results."
        )

    lines = [
        "Here are the current web results I found:",
        "",
    ]

    sources = []

    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()

        lines.append(f"{index}. {title or 'Result'}")

        if snippet:
            lines.append(snippet)

        if url:
            lines.append(url)

        lines.append("")

        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

    return {
        "output": "\n".join(lines).strip(),
        "sources": sources,
        "tool_name": "web_search",
    }


def _try_builtin_tool_execution(step):
    capability = str(
        getattr(step, "capability", "")
        or ""
    ).strip().casefold()

    if capability in {
        "calculation",
        "mathematics",
        "arithmetic",
    }:
        return _calculator_tool_result(
            getattr(step, "task", "")
        )

    if capability in {
        "web_search",
        "internet_search",
        "current_information",
    }:
        return _web_search_tool_result(
            getattr(step, "task", "")
        )

    return None


def run_task(step, runtime_input, sources, deadline):
    try:
        builtin_result = _try_builtin_tool_execution(
            step
        )
    except Exception as exc:
        return outcome(
            step,
            "failed",
            f"Tool execution failed: {exc}",
        )

    if builtin_result is not None:
        return outcome(
            step,
            "success",
            result=builtin_result,
        )

    for attempt in range(settings.MAX_ORCHESTRATION_RETRIES + 1):
        if time.monotonic() >= deadline:
            return outcome(step, "failed", "Task deadline exceeded")
        result = agent_runtime.execute(**runtime_input)
        if result.get("status") == "success" and not result.get("error"):
            result_payload = {
                "output": result.get("output"),
                "sources": sources,
            }

            if str(step.step_id).startswith("gap_worker_"):
                result_payload["worker_type"] = (
                    "persistent"
                    if step.agent_id
                    else "temporary"
                )
                result_payload["worker_name"] = (
                    step.agent_name
                    or (
                        "Auto-created Agent"
                        if step.agent_id
                        else "Temporary Worker"
                    )
                )

            return outcome(
                step,
                "success",
                result=result_payload,
            )
        if not result.get("retryable") or attempt == settings.MAX_ORCHESTRATION_RETRIES:
            runtime_error = str(result.get("error") or "Agent execution failed")
            return outcome(step, "failed", runtime_error)
        delay = min(0.25 * 2 ** attempt, max(0, deadline - time.monotonic()))
        time.sleep(delay)
    return outcome(step, "failed", "Agent execution failed")


def execute_wave(state):
    plan = ExecutionPlan.model_validate(state["plan"])
    state["iteration"] = state.get("iteration", 0) + 1
    results = state.setdefault("step_results", {})
    completed = state.setdefault("completed", [])
    deadline = state.get("deadline") or time.monotonic() + settings.MAX_ORCHESTRATION_EXECUTION_TIME
    pending = {}
    for step in plan.steps:
        if step.step_id in results:
            continue
        if time.monotonic() >= deadline:
            results[step.step_id] = outcome(step, "failed", "Orchestration deadline exceeded")
            completed.append(step.step_id)
            continue
        if any(dep not in results for dep in step.depends_on):
            continue
        if any(results[dep]["status"] != "success" for dep in step.depends_on):
            results[step.step_id] = outcome(step, "skipped", "Required dependency failed")
            completed.append(step.step_id)
            continue
        if step.condition_on and step.condition_keyword:
            previous = results[step.condition_on].get("result") or {}
            if step.condition_keyword.casefold() not in str(previous.get("output", "")).casefold():
                results[step.step_id] = outcome(step, "skipped", "Condition not met")
                completed.append(step.step_id)
                continue
        if len(pending) >= settings.MAX_CONCURRENT_TASKS:
            break
        try:
            runtime_input, sources = prepare_task(
                state["db"],
                step,
                state,
            )
        except Exception as exc:
            logger.exception(
                "manager.task_preparation_failed "
                "execution_id=%s step_id=%s agent_id=%s error=%s",
                state.get("execution_id"),
                step.step_id,
                step.agent_id,
                exc,
            )
            safe_error = str(exc).strip() or exc.__class__.__name__

            if isinstance(exc, RAGConfigurationError):
                error_message = safe_error
            else:
                error_message = f"Task preparation failed: {safe_error}"

            results[step.step_id] = outcome(
                step,
                "failed",
                error_message,
            )
            completed.append(step.step_id)
            continue

        try:
            task_deadline = min(
                deadline,
                time.monotonic()
                + settings.EXECUTION_TIMEOUT_SECONDS,
            )
            future = submit_bounded(
                lambda s=step, i=runtime_input, src=sources, d=task_deadline:
                    run_task(s, i, src, d)
            )
            pending[future] = (
                step,
                task_deadline,
            )
        except Exception as exc:
            logger.exception(
                "manager.task_admission_failed "
                "execution_id=%s step_id=%s agent_id=%s error=%s",
                state.get("execution_id"),
                step.step_id,
                step.agent_id,
                exc,
            )
            results[step.step_id] = outcome(
                step,
                "failed",
                "Task admission failed",
            )
            completed.append(step.step_id)
    while pending:
        now = time.monotonic()
        wait(pending, timeout=max(0, min(d for _, d in pending.values()) - now), return_when="FIRST_COMPLETED")
        for future, (step, task_deadline) in list(pending.items()):
            if future.done():
                try:
                    results[step.step_id] = future.result()
                except Exception:
                    results[step.step_id] = outcome(step, "failed", "Agent execution failed")
            elif time.monotonic() >= task_deadline:
                future.cancel()
                results[step.step_id] = outcome(step, "failed", "Task timed out; external work may still be running")
            else:
                continue
            completed.append(step.step_id)
            del pending[future]
    return state
