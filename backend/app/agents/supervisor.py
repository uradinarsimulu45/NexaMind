def supervisor_agent(state):
    """
    Supervisor Agent

    Decides the order in which the agents should run.
    """

    if not state.get("documents"):
        return "retrieve"

    if not state.get("answer"):
        return "generate"

    return "end"