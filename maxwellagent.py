import os
import requests
import time
from collections import Counter
import re

# Given API configuration from starter guide
API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

# For using a single shared HTTP session for all requests
session = requests.Session()

# Safety cap so no answers are too long
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

def decodingparamsdecider(question: str) -> dict:
    # Chooses temperature and max_tokens based on question text
    # This is one of my inference time techniques
    q = question.lower()

    # Checks for true or false questions
    if "true or false" in q or "t/f" in q or "true/false" in q or "yes or no" in q:
        return {"temperature": 0.0, "max_tokens": 32}
    
    # Checks for MCQ
    if "options:" in q:
        return {"temperature": 0.1, "max_tokens": 64}
    
    # Checks for math questions
    if any(token in q for token in ["how many", "calculate", "how muchh", "what is the total"]):
        return {"temperature": 0.0, "max_tokens": 64}
    
    #If it finds nothing, defaults at a higher temperature
    return {"temperature": 0.1, "max_tokens": 128}

def get_final_answer(text: str, question_text: str) -> str:
    # Process the raw model response and make it shorter
    if not text:
        return ""

    q_lower = question_text.lower()

    # Take the first non empty line of the LLM response
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ans = lines[0] if lines else ""

    # Strip the answer is
    if ans.lower().startswith("the answer is:"):
        ans = ans.split(":", 1)[1].strip()

    # Strip other common prefixes or markers
    for prefix in ("Answer:", "Ans:", "Final answer:", "Final:", "- ", "* ", "• "):
        if ans.lower().startswith(prefix.lower()):
            ans = ans[len(prefix):].strip()

    # This is for multiple choice because I noticed a lot of {} or ()'s
    if "options:" in q_lower:
        stripped = ans.strip()

        if (
            len(stripped) == 3
            and stripped[0] in "([{" 
            and stripped[2] in ")]}"
            and stripped[1] in "ABCDE"
        ):
            ans = stripped[1]

    # For any answer that begins with A-E, the first char is the MC letter
    elif ans and ans[0] in "ABCDE":
        ans = ans[0]

    # Try to infer MC letter from the patterns in the text
    else:
        for ch in "ABCDE":
            token = f"({ch})"
            if token in ans:
                ans = ch
                break
    
    # For handling numeric question and trying to pull the last number token
    if any(tok in q_lower for tok in ["how many", "how much", "calculate", "total", "miles", "articles", "ounces", "dollars", "$"]):
        matches = re.findall(r"\$?\d+(?:\.\d+)?%?", ans)
        if matches:
            ans = matches[-1]

    # Handles yes and no
    low = ans.lower()
    if low.startswith("yes"):
        ans = "Yes"
    elif low.startswith("no"):
        ans = "No"

    # Final trimming
    ans = ans.strip()
    if len(ans) > MAX_OUTPUT_CHARS:
        ans = ans[:MAX_OUTPUT_CHARS]

    return ans

def single_reasoning_call(question_text: str, system_msg: str) -> str:
    # Perform a single call to the model for one question
    # Uses decodingparamsdecier() to pick temperature/max_tokens
    # Builds a stricter prompt instruction to the model and extracts it with get_final_answer()

    params = decodingparamsdecider(question_text)

    # Prompt to make the model only respond with final answer
    reasoning_prompt = (
        f"Question:\n{question_text}\n\n"
        "Answer with ONLY the final answer.\n"
        "- Do NOT show your work.\n"
        "- If it's a math problem, only give the final answer.\n"
        "- Do NOT explain.\n"
        "- Do NOT repeat the question.\n"
        "- If the question has options (A, B, C, D, E), answer with ONLY the letter."
    )

    # Call the actual endpoint
    result = call_model_chat_completions(
        prompt=reasoning_prompt,
        system=system_msg,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
    )

    # If failure, return empty string
    if not result["ok"]:
        return ""

    raw_text = (result["text"] or "").strip()
    # Return short final answer
    return get_final_answer(raw_text, question_text)

def agent_loop(question_text: str) -> str:
    system_msg = (
    "You are a careful, precise question-answering system. "
    "You must respond with ONLY the final short answer. "
    "Do NOT write full sentences. Do NOT explain. "
    "Your reply must be a single word, number, or letter when possible."
    )

    answers = []

    # First call
    a1 = single_reasoning_call(question_text, system_msg)
    if a1:
        answers.append(a1)

    # Second call
    a2 = single_reasoning_call(question_text, system_msg)
    if a2:
        answers.append(a2)

    # If the first calls agree, return
    if len(answers) == 2 and answers[0] == answers[1]:
        return answers[0]

    # Third call as a tie breaker
    a3 = single_reasoning_call(question_text, system_msg)
    if a3:
        answers.append(a3)

    # Get rid of any empty answers
    answers = [a for a in answers if a]

    # If everything failed, return an empty string
    if not answers:
        return ""

    # Majority vote over the 3 calls
    counts = Counter(answers)
    best_answer, _ = counts.most_common(1)[0]

    # Final trimming for safety
    best_answer = best_answer.strip()
    if len(best_answer) > MAX_OUTPUT_CHARS:
        best_answer = best_answer[:MAX_OUTPUT_CHARS]

    return best_answer

