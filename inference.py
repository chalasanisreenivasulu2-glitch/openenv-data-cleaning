"""
inference.py — Baseline inference script for DataCleaning OpenEnv.

Uses the OpenAI client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API key (used as openai api_key)

Run:
  python inference.py
"""

import json
import os
import sys
import requests

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
# Credentials are read from environment variables.
# The script uses the OpenAI client which works with any OpenAI-compatible endpoint.
# Judges: set API_BASE_URL, MODEL_NAME, HF_TOKEN to your preferred provider.
# Default: Groq (free tier) with llama-3.1-8b-instant
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

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

IMPORTANT RULES:
- The JSON key MUST be "operation" — never "action" or anything else
- For date columns: use standardize_format with format_type=date -> converts to YYYY-MM-DD
- For phone columns: use standardize_format with format_type=phone -> converts to (XXX)XXX-XXXX
- For gender columns: use standardize_format with format_type=category_gender -> normalizes to M or F. Do NOT reverse this afterward.
- NEVER use fill_missing or drop_outliers on text columns like dates, phones, or gender
- NEVER use replace_value to undo a standardize_format you already applied
- Once a column is standardized, move on to the next issue
- When all issues are resolved, issue finish

Always respond with ONLY a valid JSON object. No explanation, no markdown, just raw JSON.
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
    print(f"\n{'='*50}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*50}")

    obs = env_reset(task_id)
    history = []  # fresh history per task — no leakage between tasks
    final_reward = 0.0

    # Build task-specific instructions so agent knows exactly what to do
    task_instructions = {
        "easy": (
            "Step 1: remove_duplicates\n"
            "Step 2: fill_missing on salary with strategy=mean\n"
            "Step 3: finish"
        ),
        "medium": (
            "Step 1: standardize_format on gender with format_type=category_gender (normalizes to M/F)\n"
            "Step 2: standardize_format on dob with format_type=date (normalizes to YYYY-MM-DD)\n"
            "Step 3: standardize_format on phone with format_type=phone (normalizes to (XXX)XXX-XXXX)\n"
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
        f"Only use columns that exist in the dataset. Issue one JSON action per response."
    )
    history.append({"role": "user", "content": briefing})
    history.append({"role": "assistant", "content": '{"operation": "remove_duplicates"}'
        if obs['duplicate_count'] > 0 else
        '{"operation": "standardize_format", "column": "gender", "format_type": "category_gender"}'
        if task_id == "medium" else
        '{"operation": "fill_missing", "column": "quantity", "strategy": "mean"}'
    })

    for step_num in range(100):  # hard cap
        # Build message
        # Compact observation to save tokens
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

        # Keep only last 6 messages to avoid token limit (3 exchanges)
        trimmed_history = history[-6:] if len(history) > 6 else history

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + trimmed_history,
            max_tokens=150,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        print(f"Step {step_num+1} | Agent: {raw}")

        # Parse action
        try:
            # Strip markdown code fences if present
            clean = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(clean)
            # Auto-correct common LLM mistakes: "action" -> "operation"
            if "action" in action and "operation" not in action:
                action["operation"] = action.pop("action")
            if "command" in action and "operation" not in action:
                action["operation"] = action.pop("command")
            if "type" in action and "operation" not in action:
                action["operation"] = action.pop("type")
        except json.JSONDecodeError:
            print(f"  [WARN] Could not parse action, finishing.")
            action = {"operation": "finish"}

        history.append({"role": "assistant", "content": raw})

        result = env_step(task_id, action)
        obs = result["observation"]
        final_reward = result["reward"]
        done = result["done"]
        error = result["info"].get("error")

        print(f"         Reward: {final_reward:.3f} | Done: {done}")
        if error:
            print(f"         Error: {error}")
            # Tell agent about the error so it can try something different
            history.append({"role": "user", "content": f"Error: {error}. Try a different operation or different parameters."})

        if done:
            break

        # Early exit if perfect score
        if final_reward >= 1.0:
            print(f"         [Perfect score reached, finishing early]")
            env_step(task_id, {"operation": "finish"})
            break

        # If no issues remain, force finish
        if not obs["issues"] and obs["duplicate_count"] == 0 and all(v == 0 for v in obs["missing_counts"].values()):
            print(f"         [No issues remain, forcing finish]")
            env_step(task_id, {"operation": "finish"})
            break

        # Also tell the agent the current reward so it knows when to stop
        obs["current_reward"] = final_reward

    print(f"\n  Final score for '{task_id}': {final_reward:.4f}")
    return final_reward


def main():
    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] Task '{task_id}' failed: {e}")
            scores[task_id] = 0.0

    print(f"\n{'='*50}")
    print("  FINAL SCORES")
    print(f"{'='*50}")
    for task_id, score in scores.items():
        print(f"  {task_id:8s}: {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':8s}: {avg:.4f}")
    print(f"{'='*50}")

    # Write scores to file for CI
    with open("scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print("\nScores written to scores.json")


if __name__ == "__main__":
    main()
