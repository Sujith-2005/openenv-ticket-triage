from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from models import Action, StepResponse, State, Observation
from environment import SupportTriageEnv
from tasks import TASKS
import uvicorn

app = FastAPI(title="OpenEnv Ticket Triage", description="A simulated environment for routing customer support tickets.")

# Simple global state for standard evaluation
current_env = None

@app.get("/")
def read_root():
    return {"status": "ok"}

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"

@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    global current_env
    task_to_load = "task_1_easy"
    if req and req.task_id:
        task_to_load = req.task_id
        
    if task_to_load not in TASKS:
        raise HTTPException(status_code=400, detail=f"Task {task_to_load} not found. Available tasks: {list(TASKS.keys())}")
        
    current_env = SupportTriageEnv(task_id=task_to_load)
    return current_env.reset()

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global current_env
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs, reward, done, info = current_env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=State)
def state():
    if not current_env:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return current_env.get_state()

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": list(TASKS.keys()),
        "action_schema": Action.model_json_schema()
    }

@app.get("/grader")
def get_grader():
    if not current_env:
         raise HTTPException(status_code=400, detail="Environment not initialized.")
    return {
        "score": current_env.get_score(),
        "done": current_env.done
    }

@app.post("/baseline")
def run_baseline_endpoint():
    from inference import run_baseline
    try:
        scores = run_baseline()
        return {"baseline_scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline failed: {str(e)}")

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
