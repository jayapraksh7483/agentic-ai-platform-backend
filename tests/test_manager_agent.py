"""
Manager Agent tests, updated for the current architecture:
    - classify_request() routes general/greeting messages to a direct
      LLM response, bypassing capability extraction and agent proposal
      entirely.
    - extract_capabilities() shows the LLM a NUMBERED list of the exact
      registered capability strings and gets back INDICES, never
      free-text capability names.
    - When nothing in the registry matches, the Manager drafts a
      ProposedAgentSpec and returns status "pending_agent_approval"
      instead of "failed" -- the caller must approve it via
      approve_pending_agent() before anything is created.

Run with:
    pytest tests/test_manager_agent.py -v
"""
import json
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import core.database as database_module


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    database_module.engine = engine
    SessionLocal = sessionmaker(bind=engine)
    database_module.SessionLocal = SessionLocal

    from models.agent import Agent, AgentVersion, AgentCapability
    from models.orchestration import OrchestrationExecution, OrchestrationStep
    from core.database import Base

    Base.metadata.create_all(
        bind=engine,
        tables=[
            Agent.__table__, AgentVersion.__table__, AgentCapability.__table__,
            OrchestrationExecution.__table__, OrchestrationStep.__table__,
        ],
    )

    session = SessionLocal()
    yield session
    session.close()


def _register_agent(db, name, capability):
    from models.agent import Agent, AgentVersion, AgentCapability, AgentStatus

    agent = Agent(
        name=name, description=name, system_prompt=f"You are {name}.",
        provider="gemini", model="gemini-3.6-flash",
        status=AgentStatus.ACTIVE, current_version=1,
    )
    db.add(agent)
    db.flush()
    db.add(AgentCapability(agent_id=agent.id, capability_name=capability))
    db.add(AgentVersion(agent_id=agent.id, version_number=1, system_prompt=agent.system_prompt))
    db.commit()
    db.refresh(agent)
    return agent


class FakeLLMClient:
    """Deterministic stand-in for services.llm_service's LLM client."""

    def generate(self, system_prompt, user_input, model_name, temperature=None):
        # --- Stage 0: general vs agent_task classification ---
        if "Classify the user's request" in system_prompt:
            text = user_input.lower().strip()
            greeting_phrases = ("hello", "hi,", "hi ", "hey", "thanks", "how are you", "what can you do")
            if text.startswith(greeting_phrases) or any(p in text for p in ("thanks", "how are you", "what can you do")):
                return '{"request_type": "general"}'
            return '{"request_type": "agent_task"}'

        # --- Direct conversational response (general requests) ---
        if "Answer simple conversational requests directly" in system_prompt:
            return "Hello! I can understand requests, find the right registered agent, and run them for you."

        # --- Result synthesis ---
        if "Step results" in user_input:
            data = json.loads(user_input.split("Step results (JSON):\n", 1)[1])
            return "Result: " + "; ".join(str(d.get("result") or d.get("error")) for d in data)

        # --- Capability extraction (index-based) ---
        if "selected_capabilities" in system_prompt:
            lines = [l for l in user_input.split("\n") if l.strip().startswith("[")]
            text = user_input.lower()

            def find(keyword):
                for line in lines:
                    if keyword in line.lower():
                        return line.split("]")[0].strip("[")
                return None

            if "customer 101" in text and "outstanding" in text:
                idx_customer, idx_calc = find("customer"), find("calculat")
                if idx_customer is not None and idx_calc is not None:
                    return json.dumps({
                        "selected_capabilities": [int(idx_customer), int(idx_calc)],
                        "execution_mode": "sequential", "condition_keyword": None,
                        "unmatched_description": None,
                    })
            if "customer details" in text and "policy" in text:
                idx_customer, idx_rag = find("customer"), find("knowledge")
                if idx_customer is not None and idx_rag is not None:
                    return json.dumps({
                        "selected_capabilities": [int(idx_customer), int(idx_rag)],
                        "execution_mode": "parallel", "condition_keyword": None,
                        "unmatched_description": None,
                    })
            for keyword in ("calculat", "customer", "weather", "knowledge", "general", "handling"):
                idx = find(keyword)
                if idx is not None:
                    return json.dumps({
                        "selected_capabilities": [int(idx)],
                        "execution_mode": "sequential", "condition_keyword": None,
                        "unmatched_description": None,
                    })

            return json.dumps({
                "selected_capabilities": [],
                "unmatched_description": "handling this kind of request",
                "execution_mode": "sequential", "condition_keyword": None,
            })

        # --- New-agent proposal ---
        if "Agent Designer" in system_prompt:
            return json.dumps({
                "name": "General Purpose Agent",
                "description": "Handles requests outside the current registry.",
                "capabilities": ["handling this kind of request"],
                "system_prompt": "You are a helpful general-purpose agent.",
                "provider": "gemini", "model": None,
                "reason": "No registered agent currently covers this domain.",
            })

        return "[unhandled fake prompt]"

    def generate_with_tool_calls(self, system_prompt, messages, model_name, tool_specs, temperature=None):
        prompt = system_prompt.lower()
        if "customer" in prompt:
            return {"content": "Customer 101: outstanding balance 12500", "tool_calls": []}
        if "calculat" in prompt:
            return {"content": "12500", "tool_calls": []}
        if "knowledge" in prompt or "rag" in prompt:
            return {"content": "Policy doc: standard coverage terms.", "tool_calls": []}
        if "weather" in prompt:
            return {"content": "Sunny, 25C.", "tool_calls": []}
        return {"content": "ok", "tool_calls": []}


@pytest.fixture(autouse=True)
def fake_llm():
    import services.llm_service as llm_service
    llm_service._client_cache["gemini"] = FakeLLMClient()
    yield
    llm_service._client_cache.pop("gemini", None)


def test_greeting_gets_direct_response_not_agent_proposal(db_session):
    """Regression test: greetings/general chat must NOT trigger a new-agent
    proposal -- they should be answered directly by the Manager."""
    from manager import service as manager_service

    response = manager_service.orchestrate(db_session, "Hello, how are you?")

    assert response.status == "success"
    assert "create" not in (response.result or "").lower()
    assert "should i go ahead" not in (response.result or "").lower()


def test_single_agent_calculator(db_session):
    from manager import service as manager_service

    _register_agent(db_session, "calculator-agent", "arithmetic calculations")

    response = manager_service.orchestrate(db_session, "What is 25 multiplied by 16?")

    assert response.status == "success"
    assert response.error is None


def test_sequential_multi_agent(db_session):
    from manager import service as manager_service

    _register_agent(db_session, "customer-agent", "customer lookup")
    _register_agent(db_session, "calculator-agent", "arithmetic calculations")

    response = manager_service.orchestrate(
        db_session, "Find customer 101 and calculate his outstanding amount."
    )

    assert response.status == "success"
    assert "12500" in response.result

    status = manager_service.get_execution(db_session, response.execution_id)
    assert [s.step_key for s in status.steps] == ["step_1", "step_2"]
    assert status.steps[1].depends_on == ["step_1"]


def test_parallel_execution(db_session):
    from manager import service as manager_service

    _register_agent(db_session, "customer-agent", "customer lookup")
    _register_agent(db_session, "rag-agent", "knowledge search")

    response = manager_service.orchestrate(
        db_session, "Get customer details and search the policy documents."
    )

    assert response.status == "success"
    status = manager_service.get_execution(db_session, response.execution_id)
    for step in status.steps:
        assert step.depends_on in (None, [])


def test_new_agent_discovered_without_code_change(db_session):
    from manager import service as manager_service

    _register_agent(db_session, "weather-agent", "weather lookup")

    response = manager_service.orchestrate(db_session, "What is the weather?")

    assert response.status == "success"


def test_unsupported_request_drafts_proposal_not_hard_failure(db_session):
    """Nothing registered can help -> Manager proposes a new agent and
    waits for approval, rather than fabricating an agent or a result."""
    from manager import service as manager_service

    response = manager_service.orchestrate(db_session, "Do something completely unsupported.")

    assert response.status == "pending_agent_approval"
    assert response.error is None


def test_approve_creates_real_agent_from_proposal_and_reruns(db_session):
    """The persisted proposal must be the ACTUAL drafted spec (name,
    capabilities, system_prompt) -- not a blank/generic agent -- and
    approving it should let the original request succeed."""
    from manager import service as manager_service
    from models.agent import Agent
    from models.orchestration import OrchestrationExecution

    first = manager_service.orchestrate(db_session, "Do something completely unsupported.")
    assert first.status == "pending_agent_approval"

    row = db_session.query(OrchestrationExecution).get(first.execution_id)
    assert row.proposed_agent is not None
    assert row.proposed_agent["name"] == "General Purpose Agent"

    approved = manager_service.approve_pending_agent(db_session, first.execution_id, approved=True)
    assert approved.status == "success"

    created = db_session.query(Agent).filter(Agent.name == "General Purpose Agent").first()
    assert created is not None
    assert created.system_prompt == "You are a helpful general-purpose agent."


def test_decline_proposal_does_not_create_agent(db_session):
    from manager import service as manager_service
    from models.agent import Agent

    first = manager_service.orchestrate(db_session, "Do something completely unsupported.")
    assert first.status == "pending_agent_approval"

    declined = manager_service.approve_pending_agent(db_session, first.execution_id, approved=False)

    assert declined.status == "failed"
    assert "declined" in declined.error.lower()
    assert db_session.query(Agent).count() == 0


def test_no_hard_coded_routing_in_source():
    import inspect
    from manager import planner

    source = inspect.getsource(planner)
    assert "calculator-agent" not in source
    assert "customer-agent" not in source
    assert "if capability ==" not in source