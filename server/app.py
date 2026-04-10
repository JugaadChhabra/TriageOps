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
