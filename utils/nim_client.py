# nim_client.py
from typing import Dict
import requests

import utils.config as config


def _nim_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }


def nim_chat_completion(system_prompt: str, user_prompt: str) -> str:
    """
    Call NIM chat completions (OpenAI-compatible: POST /v1/chat/completions).
    """
    url = f"{config.NVIDIA_BASE_URL}/chat/completions"
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    resp = requests.post(url, headers=_nim_headers(), json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
