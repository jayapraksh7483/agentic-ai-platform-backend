from typing import List

from sqlalchemy.orm import Session

from core.config import settings

from models.agent import (
    Agent,
    AgentCapability,
    AgentVersion,
)

from schemas.agent import (
    AgentCreate,
    ModelConfig,
)

from services.agent_service import create_agent


DEFAULT_DOCUMENT_READER_NAME = "Document Reader"
DEFAULT_WEB_SEARCH_NAME = "Web Search"
DEFAULT_CALCULATOR_NAME = "Calculator"
DEFAULT_GENERAL_ASSISTANT_NAME = "General Assistant"


# Always use the platform-configured Gemini default model.
DEFAULT_GEMINI_MODEL = settings.GEMINI_DEFAULT_MODEL


DEFAULT_AGENT_NAMES = {
    DEFAULT_DOCUMENT_READER_NAME,
    DEFAULT_WEB_SEARCH_NAME,
    DEFAULT_CALCULATOR_NAME,
    DEFAULT_GENERAL_ASSISTANT_NAME,
}


def _get_user_agents(
    db: Session,
    user_id: int,
) -> dict:
    """
    Return the user's agents indexed by name.
    """

    agents = (
        db.query(Agent)
        .filter(
            Agent.created_by == user_id
        )
        .all()
    )

    return {
        agent.name: agent
        for agent in agents
    }


def _build_default_agent_definitions() -> List[dict]:
    """
    Return the current platform definitions for
    all default agents.
    """

    return [
        {
            "name": DEFAULT_DOCUMENT_READER_NAME,

            "description": (
                "Reads and answers questions about the "
                "user's uploaded documents using "
                "retrieval-augmented generation. "
                "Document context is selected dynamically "
                "from the authenticated user's knowledge bases."
            ),

            "system_prompt": (
                "You are a Document Reader agent. "
                "Your primary purpose is to answer questions "
                "using the authenticated user's uploaded "
                "documents and retrieved document context. "
                "Use only the document context provided by "
                "the platform for document-related answers. "
                "Do not invent information that is not supported "
                "by the retrieved document context. "
                "If the required information is not available "
                "in the provided documents, clearly say that "
                "it could not be found. "
                "Prefer document evidence over general knowledge."
            ),

            "capabilities": [
                "document_search",
                "document_question_answering",
                "rag",
                "document_retrieval",
            ],

            "is_rag": True,
            "knowledge_base_id": None,
        },

        {
            "name": DEFAULT_WEB_SEARCH_NAME,

            "description": (
                "Searches the web for current information "
                "using the backend-controlled DuckDuckGo "
                "search tool."
            ),

            "system_prompt": (
                "You are a Web Search agent. "
                "Use the available backend web-search tool "
                "when the user needs current, external, or "
                "web-based information. "
                "Do not claim that you searched the web unless "
                "the search tool actually returned results. "
                "Use the returned search results and sources "
                "to form your answer. "
                "Do not fabricate URLs, sources, or search results."
            ),

            "capabilities": [
                "web_search",
                "internet_search",
                "current_information",
            ],

            "is_rag": False,
            "knowledge_base_id": None,
        },

        {
            "name": DEFAULT_CALCULATOR_NAME,

            "description": (
                "Performs mathematical calculations and "
                "numerical reasoning."
            ),

            "system_prompt": (
                "You are a Calculator agent. "
                "Solve mathematical and numerical problems accurately. "
                "Use the available calculator capability when appropriate. "
                "Show the important calculation steps when useful. "
                "Do not invent numerical results."
            ),

            "capabilities": [
                "calculation",
                "mathematics",
                "arithmetic",
            ],

            "is_rag": False,
            "knowledge_base_id": None,
        },

        {
            "name": DEFAULT_GENERAL_ASSISTANT_NAME,

            "description": (
                "Handles general questions and everyday conversations."
            ),

            "system_prompt": (
                "You are a helpful general-purpose AI assistant. "
                "Answer the user's questions clearly, accurately, "
                "and concisely. "
                "When a specialized registered agent is more suitable "
                "for a request, allow the platform manager to route "
                "the request to that specialized agent."
            ),

            "capabilities": [
                "general_conversation",
                "question_answering",
                "general_assistance",
            ],

            "is_rag": False,
            "knowledge_base_id": None,
        },
    ]


def _synchronize_existing_agent(
    db: Session,
    agent: Agent,
    definition: dict,
) -> Agent:
    """
    Synchronize an existing platform default agent
    with the current platform definition.

    The existing agent ID is preserved.
    """

    changed_for_version = False

    # --------------------------------------------------------------
    # Basic configuration
    # --------------------------------------------------------------

    if agent.description != definition["description"]:
        agent.description = definition["description"]

    if agent.system_prompt != definition["system_prompt"]:
        agent.system_prompt = definition["system_prompt"]
        changed_for_version = True

    # --------------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------------

    if (
        agent.provider != "gemini"
        or agent.model != DEFAULT_GEMINI_MODEL
    ):
        agent.provider = "gemini"
        agent.model = DEFAULT_GEMINI_MODEL
        changed_for_version = True

    # --------------------------------------------------------------
    # RAG configuration
    # --------------------------------------------------------------

    agent.is_rag = definition["is_rag"]

    agent.knowledge_base_id = (
        definition["knowledge_base_id"]
    )

    # --------------------------------------------------------------
    # Default marker
    # --------------------------------------------------------------

    agent.is_default = True

    # Default agents must remain active.

    # --------------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------------

    desired_capabilities = set(
        definition["capabilities"]
    )

    existing_capabilities = {
        capability.capability_name: capability
        for capability in agent.capabilities
    }

    # Remove obsolete capabilities.
    for (
        capability_name,
        capability,
    ) in list(existing_capabilities.items()):

        if capability_name not in desired_capabilities:
            db.delete(capability)

    # Add missing capabilities.
    existing_capability_names = {
        capability.capability_name
        for capability in agent.capabilities
        if capability.capability_name
        in desired_capabilities
    }

    for capability_name in desired_capabilities:

        if (
            capability_name
            not in existing_capability_names
        ):

            db.add(
                AgentCapability(
                    agent_id=agent.id,
                    capability_name=capability_name,
                )
            )

    # --------------------------------------------------------------
    # Version snapshot
    # --------------------------------------------------------------

    if changed_for_version:

        agent.current_version += 1

        db.add(
            AgentVersion(
                agent_id=agent.id,
                version_number=agent.current_version,
                system_prompt=agent.system_prompt,
                model_config_snapshot={
                    "provider": agent.provider,
                    "model": agent.model,
                },
            )
        )

    return agent


def provision_default_agents(
    db: Session,
    user_id: int,
) -> List[Agent]:
    """
    Provision the four platform default agents for a user.

    Idempotent:
        Running this repeatedly does not create duplicates.

    Existing default agents are synchronized in place.

    Missing default agents are created.

    User-created agents remain untouched.
    """

    existing_agents = _get_user_agents(
        db=db,
        user_id=user_id,
    )

    default_agents: List[Agent] = []

    for definition in _build_default_agent_definitions():

        name = definition["name"]

        existing_agent = existing_agents.get(name)

        # ----------------------------------------------------------
        # Existing default agent
        # ----------------------------------------------------------

        if existing_agent is not None:

            # Never overwrite user configuration at login, including disabled defaults.
            default_agents.append(existing_agent)
            continue

        # ----------------------------------------------------------
        # Missing default agent
        # ----------------------------------------------------------

        agent_in = AgentCreate(
            name=name,
            description=definition["description"],
            system_prompt=definition["system_prompt"],
            capabilities=definition["capabilities"],
            input_schema=None,
            output_schema=None,

            model_cfg=ModelConfig(
                provider="gemini",
                model=DEFAULT_GEMINI_MODEL,
            ),

            is_rag=definition["is_rag"],
            knowledge_base_id=definition[
                "knowledge_base_id"
            ],
        )

        agent = create_agent(
            db=db,
            agent_in=agent_in,
            created_by=user_id,
            is_default=True,
        )

        default_agents.append(agent)

        existing_agents[name] = agent

    db.commit()

    for agent in default_agents:
        db.refresh(agent)

    return default_agents
