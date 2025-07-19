from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .tasks import run_orchestrator
from .logging import logger

app = FastAPI(title="AgentFlow Orchestrator")

class TaskIn(BaseModel):
    task_description: str

@app.post("/orchestrate")
async def orchestrate(req: TaskIn):
    try:
        result = run_orchestrator(req.task_description)
        logger.info(f"Orchestration complete for {req.task_description}")
        return {"result": result}
    except Exception as exc:
        logger.error(f"Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

