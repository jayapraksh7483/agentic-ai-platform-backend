"""Public decision projection of the existing intent, plan and failure contracts."""


def decision_for_result(result):
    if result.get("status") == "failed":
        return "CONTROLLED_FAILURE"
    proposal = result.get("proposed_agent") or {}
    if proposal.get("is_rag"):
        return "RAG_AGENT_CREATION"
    if result.get("capability_gap"):
        return "CAPABILITY_GAP"
    if result.get("failure_code") in {"AGENT_ALREADY_EXISTS", "AMBIGUOUS_APPROVAL"}:
        return "CLARIFICATION_REQUIRED"
    if result.get("intent") in {"AGENT_CREATE", "AGENT_UPDATE", "AGENT_DELETE", "AGENT_QUERY"}:
        return "AGENT_LIFECYCLE"
    steps = (result.get("plan") or {}).get("steps") or []
    if len(steps) > 1:
        return "MULTI_AGENT_EXECUTION"
    if steps:
        return "TOOL_EXECUTION" if steps[0].get("tool_name") else "SINGLE_AGENT_EXECUTION"
    return "DIRECT_RESPONSE"
