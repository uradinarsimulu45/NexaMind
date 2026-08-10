def supervisor_agent(state):
    """
    Supervisor Agent
    Decides which agent should run next.
    """

    if not state.get("documents"):
        return {"next": "retrieve"}

    if not state.get("answer"):
        return {"next": "generate"}

    return {"next": "end"}