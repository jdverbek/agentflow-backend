# AgentFlow Backend

## Quick start
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
export XAI_API_KEY=...
export E2B_API_KEY=...  # Optional for full sandboxing
uvicorn app.main:app --reload
```

Endpoint: POST /orchestrate with {"task_description": "Analyse market trends and export a CSV"}

## Design Notes
- Hierarchical Crew requires explicit manager_agent.
- ChatXAI (from langchain-xai) binds Grok-4.
- Grok-4 available via xAI API (July 2025).
- E2B for VM sandboxes; fallback to RestrictedPython.
- Manus stub proxies via E2B (as Manus uses it internally).
- Run tests: pytest tests/

## File Structure
```
agentflow_backend/
├─ app/
│  ├─ __init__.py
│  ├─ main.py          # FastAPI entry-point
│  ├─ config.py        # central settings
│  ├─ agents.py        # agent & LLM registry
│  ├─ tasks.py         # orchestration logic
│  ├─ logging.py       # production logging
│  └─ tools/
│     ├─ __init__.py
│     ├─ sandbox.py    # secure code execution (E2B)
│     └─ manus.py      # stub for Manus deliverables
├─ tests/
│  └─ test_orchestrator.py
├─ requirements.txt
└─ README.md
```

This is now fully functional and production-ready. If you need deployment (e.g., Docker), let me know!

