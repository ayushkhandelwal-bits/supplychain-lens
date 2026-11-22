"""Optional local LLM via Ollama (ADR-0004). Never required; never sees anything but tool outputs.

Enable locally:
    ollama pull qwen2.5:3b
    set OLLAMA_HOST=http://localhost:11434      (PowerShell: $env:OLLAMA_HOST="http://localhost:11434")
    set SCL_OLLAMA_MODEL=qwen2.5:3b             (optional)

Guardrail: every number in the LLM's answer must appear in the facts it was given; otherwise the
deterministic template answer is used instead.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*\.?\d*%?[kMB]?")


def available() -> bool:
    host = os.environ.get("OLLAMA_HOST")
    if not host:
        return False
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _norm(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").rstrip("%").rstrip(".")


def numbers_grounded(answer: str, facts: str) -> bool:
    """True if every number in ``answer`` also appears in ``facts`` (format-insensitive)."""
    allowed = {_norm(t) for t in NUMBER.findall(facts)}
    for tok in NUMBER.findall(answer):
        n = _norm(tok)
        if n and n not in allowed and n not in {"1", "2", "3", "4", "5", "14", "28"}:  # list numbering / horizons
            return False
    return True


def rewrite(facts: str, question: str, timeout: int = 60) -> str | None:
    """Ask the local model to answer using ONLY the facts. Returns None if unavailable or ungrounded."""
    if not available():
        return None
    model = os.environ.get("SCL_OLLAMA_MODEL", "qwen2.5:3b")
    prompt = ("You are a supply-chain analyst. Answer the question in 3-5 short sentences using ONLY the facts below. "
              "Copy numbers exactly as written; do not compute new numbers; do not add facts.\n\n"
              f"FACTS:\n{facts}\n\nQUESTION: {question}\nANSWER:")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{os.environ['OLLAMA_HOST'].rstrip('/')}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read())["response"].strip()
    except Exception:
        return None
    return text if numbers_grounded(text, facts) else None
