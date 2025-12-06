import os
import requests
import time
from collections import Counter
import re

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

# make it run in a session
session = requests.Session()

MAX_OUTPUT_CHARS = 4900

def call_model_chat_completions(
    prompt: str,
    system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 60,
) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    {
        'ok': bool,
        'text': str or None,
        'raw': dict or None,
        'status': int,
        'error': str or None,
        'headers': dict
    }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs = dict(resp.headers)

        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "ok": True,
                "text": text,
                "raw": data,
                "status": status,
                "error": None,
                "headers": hdrs,
            }
        else:
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text

            return {
                "ok": False,
                "text": None,
                "raw": None,
                "status": status,
                "error": str(err_text),
                "headers": hdrs,
            }

    except requests.RequestException as e:
        return {
            "ok": False,
            "text": None,
            "raw": None,
            "status": -1,
            "error": str(e),
            "headers": {},
        }


def agent_loop(question_text: str) -> str:
    print("Calling model...")

    system_msg = (
        "You are a careful solver. Reply ONLY with the final answer, "
        "nothing else. Do not include explanations."
    )

    result = call_model_chat_completions(
        prompt=question_text,
        system=system_msg,
        model=MODEL,
        temperature=0.0,
    )

    if not result["ok"]:
        return ""

    answer = (result["text"] or "").strip()
    return answer
