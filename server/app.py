"""
FastAPI application for the TriageOps environment.

Built on the official OpenEnv `create_app()` factory, which provides:
  - WebSocket endpoint at /ws (used by EnvClient)
  - HTTP endpoints at /reset, /step, /state
  - /health, /schema, /docs
  - Multi-session support (each WebSocket gets its own TriageEnvironment instance)

Usage:
    # Local dev
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

    # Via uv (uses [project.scripts] server entry point)
    uv run server

    # Direct
    python -m server.app
"""

from openenv.core.env_server.http_server import create_app

from models import TriageAction, TriageObservation

from .triage_environment import TriageEnvironment

# Create the FastAPI app via OpenEnv's factory.
# This wires up all the standard endpoints (/reset, /step, /state, /ws, /health, /schema, /docs)
# and handles per-session isolation. Pass the Environment *class* (not an instance)
# so the framework can instantiate one TriageEnvironment per WebSocket session.
app = create_app(
    TriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="triageops",
    max_concurrent_envs=4,
)


# Root route — HF Space's "App" tab loads `/`, so show a useful landing
# page instead of the default 404 from create_app(). All real endpoints
# (/health, /reset, /step, /state, /ws, /docs, /schema, /metadata) come
# from create_app().
@app.get("/")
def root() -> dict:
    return {
        "name": "triageops",
        "description": "AI Customer Support Ops — OpenEnv RL Environment",
        "team": "Pied Piper (Muaaz Shaikh, Mantek Singh Burn, Jugaad Chhabra)",
        "version": "1.0.0",
        "tasks": ["ticket_classification", "triage_prioritize", "full_resolution"],
        "endpoints": {
            "websocket": "/ws",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "docs": "/docs",
        },
        "client_usage": (
            "from client import TriageOpsEnv; "
            "env = await TriageOpsEnv.from_docker_image('triageops:latest')"
        ),
        "github": "https://github.com/JugaadChhabra/TriageOps",
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for direct execution.

    Run via:
        uv run server
        python -m server.app
        python -m server.app --port 8001
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
