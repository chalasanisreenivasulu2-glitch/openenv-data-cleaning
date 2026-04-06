"""
inference.py — Baseline inference script for DataCleaning OpenEnv.

Uses the OpenAI client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API endpoint (default: Groq)
  MODEL_NAME    — model identifier (default: llama-3.1-8b-instant)
  HF_TOKEN      — API key, NO default (must be set by user)
  ENV_BASE_URL  — environment URL (default: localhost)

Run:
  python inference.py
"""

import json
import os
import sys
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME have defaults (as required by spec)
# HF_TOKEN has NO default — must be provided via environment variable
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN")          # NO default — required
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set. Please set your API key.")
    sys.exit(1)

# All LLM calls use the OpenAI client configured via environment variables
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a data cleaning agent. You control a CSV dataset by issuing JSON actions.

CRITICAL: Every JSON you return MUST have the key "operation" (not "action", not "command", not "type").

Valid JSON examples:
{"operation": "remove_duplicates"}
{"operation": "fill_missing", "column": "salary", "strategy": "mean"}
{"operation": "standardize_format", "column": "dob", "format_type": "date"}
{"operation": "standardize_format", "column": "phone", "format_type": "phone"}
{"operation": "standardize_format", "column": "gender", "format_type": "category_gender"}
{"operation": "replace_value", "column": "col", "old_value": "old", "new_value": "new"}
{"operation": "drop_outliers", "column": "revenue", "threshold": 3.0}
{"operation": "finish"}

RULES:
- JSON key MUST be "operation"
- For date columns: standardize_format with format_type=date -> YYYY-MM-DD
- For phone columns: standardize_format with format_type=phone -> (XXX)XXX-XXXX
- For gender columns: standardize_format with format_type=category_gender -> M or F
- NEVER use fill_missing or drop_outliers on text columns
- When all issues are resolved, issue finish

Respond with ONLY a valid JSON object. No explanation, no markdown.
"""


def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action: dict) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        params={"task_id": task_id},
        json=action,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()


def run_task(task_id: str) -> float:
    # ── START log ────────────────────────────────────────────────────────────
    print(f"[START] task={task_id} model={MODEL_NAME} env={ENV_BASE_URL}")

    obs = env_reset(task_id)
    history = []
    final_reward = 0.0

    task_instructions = {
        "easy": (
            "Step 1: remove_duplicates\n"
            "Step 2: fill_missing on salary with strategy=mean\n"
            "Step 3: finish"
        ),
        "medium": (
            "Step 1: standardize_format on gender with format_type=category_gender\n"
            "Step 2: standardize_format on dob with format_type=date\n"
            "Step 3: standardize_format on phone with format_type=phone\n"
            "Step 4: finish"
        ),
        "hard": (
            "Step 1: remove_duplicates\n"
            "Step 2: fill_missing on quantity with strategy=mean\n"
            "Step 3: fill_missing on region with strategy=mode\n"
            "Step 4: fill_missing on salesperson with strategy=mode\n"
            "Step 5: fill_missing on product with strategy=mode\n"
            "Step 6: standardize_format on date with format_type=date\n"
            "Step 7: standardize_format on product with format_type=title\n"
            "Step 8: drop_outliers on revenue with threshold=3.0\n"
            "Step 9: finish"
        ),
    }

    briefing = (
        f"You are cleaning a '{task_id}' dataset.\n"
        f"Columns available: {obs['columns']}\n"
        f"Follow these steps exactly in order:\n"
        f"{task_instructions.get(task_id, 'Fix all issues then finish.')}\n"
        f"Issue one JSON action per response."
    )
    history.append({"role": "user", "content": briefing})
    history.append({"role": "assistant", "content":
        '{"operation": "remove_duplicates"}' if obs['duplicate_count'] > 0 else
        '{"operation": "standardize_format", "column": "gender", "format_type": "category_gender"}' if task_id == "medium" else
        '{"operation": "fill_missing", "column": "quantity", "strategy": "mean"}'
    })

    for step_num in range(100):
        missing_only = {k: v for k, v in obs['missing_counts'].items() if v > 0}
        status_msg = "STATUS: All issues resolved. Issue finish now." if (
            not obs['issues'] and obs['duplicate_count'] == 0
            and all(v == 0 for v in obs['missing_counts'].values())
        ) else ""
        obs_text = (
            f"Task:{obs['task_id']} Step:{obs['step']}/{obs['max_steps']} Reward:{final_reward:.2f}\n"
            f"Cols:{obs['columns']}\n"
            f"Duplicates:{obs['duplicate_count']} Missing:{missing_only}\n"
            f"Issues:{obs['issues']}\n"
            f"Sample:{json.dumps(obs['sample_rows'][:2])}\n"
            + (f"{status_msg}\n" if status_msg else "")
        )

        history.append({"role": "user", "content": obs_text})
        trimmed_history = history[-6:] if len(history) > 6 else history

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history,
            max_tokens=150,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        # Parse action
        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)
            if "action" in action and "operation" not in action:
                action["operation"] = action.pop("action")
            if "command" in action and "operation" not in action:
                action["operation"] = action.pop("command")
            if "type" in action and "operation" not in action:
                action["operation"] = action.pop("type")
        except json.JSONDecodeError:
            action = {"operation": "finish"}

        history.append({"role": "assistant", "content": raw})

        result = env_step(task_id, action)
        obs = result["observation"]
        final_reward = result["reward"]
        done = result["done"]
        error = result["info"].get("error")

        # ── STEP log ─────────────────────────────────────────────────────────
        print(f"[STEP] task={task_id} step={step_num+1} action={json.dumps(action)} reward={final_reward:.4f} done={done}")
        if error:
            print(f"[STEP] task={task_id} step={step_num+1} error={error}")
            history.append({"role": "user", "content": f"Error: {error}. Try different parameters."})

        if done:
            break

        if final_reward >= 1.0:
            env_step(task_id, {"operation": "finish"})
            break

        if not obs["issues"] and obs["duplicate_count"] == 0 and all(v == 0 for v in obs["missing_counts"].values()):
            env_step(task_id, {"operation": "finish"})
            break

        obs["current_reward"] = final_reward

    # ── END log ──────────────────────────────────────────────────────────────
    print(f"[END] task={task_id} final_score={final_reward:.4f}")
    return final_reward


def main():
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] task={task_id} error={e}")
            scores[task_id] = 0.0

    print(f"\n[RESULTS]")
    for task_id, score in scores.items():
        print(f"[RESULTS] {task_id}={score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"[RESULTS] average={avg:.4f}")

    with open("scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print("[RESULTS] scores written to scores.json")


if __name__ == "__main__":
    main()
