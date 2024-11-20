from typing import Any, List, Dict

def affirmative_resp(model: Any, history: List[Dict[str, str]]) -> bool:
    history[-1]["content"] += " Answer \"YES\" or \"NO\" only."

    outputs = model(history, max_new_tokens=10)
    response = outputs[0]["generated_text"][-1]["content"].upper()

    return "NO" not in response and "YES" in response