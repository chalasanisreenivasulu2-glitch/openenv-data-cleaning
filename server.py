"""
FastAPI server exposing the OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state
"""
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from models import Action, Observation
from env import DataCleaningEnv

app = FastAPI(
    title="DataCleaning OpenEnv",
    description="OpenEnv-compliant data cleaning environment",
    version="1.0.0",
)

# One environment instance per task (stateful per session)
_envs: Dict[str, DataCleaningEnv] = {
    "easy":   DataCleaningEnv("easy"),
    "medium": DataCleaningEnv("medium"),
    "hard":   DataCleaningEnv("hard"),
}


def safe_json(obj):
    """Recursively convert numpy types to Python natives."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


@app.get("/")
def root():
    return {"status": "ok", "env": "data-cleaning-openenv", "tasks": list(_envs.keys())}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = "easy"):
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    obs: Observation = _envs[task_id].reset()
    return JSONResponse(content=safe_json(obs.model_dump()))


@app.post("/step")
def step(action: Action, task_id: str = "easy"):
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    try:
        obs, reward, done, info = _envs[task_id].step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=safe_json({
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }))


@app.get("/state")
def state(task_id: str = "easy"):
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    return JSONResponse(content=safe_json(_envs[task_id].state()))


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "easy",   "name": "Remove Duplicates & Fill Nulls",  "difficulty": "easy",   "max_steps": 10},
            {"id": "medium", "name": "Format Standardization",          "difficulty": "medium", "max_steps": 20},
            {"id": "hard",   "name": "Full Multi-Issue Cleaning",        "difficulty": "hard",   "max_steps": 30},
        ]
    }