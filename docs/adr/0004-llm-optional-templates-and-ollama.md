# ADR-0004 — LLM is optional: template narration by default, Ollama locally

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

NEXUS-SCCT requires paid LLM APIs: Claude Sonnet as primary and GPT-4 Turbo as fallback for all 8 agents, plus a browser-side copilot call to `api.anthropic.com` with no API key (it works only inside the Claude.ai artifact sandbox). Our budget is ₹0, and the public demo must work for any visitor.

A free host can't run a local LLM: model weights are gigabytes and CPU inference is slow.

## Decision

Three narration levels, selected automatically:

| Level | When | How |
|---|---|---|
| 1. Templates (default) | Always available; used on the public site | Deterministic text built from tool outputs (e.g. *"Lead time for SUP-0012 rose from 41 to 52 days (P90)…"*) |
| 2. Ollama (local) | `OLLAMA_HOST` is set and reachable | A small open model (e.g. `qwen2.5:3b` or `llama3.2:3b`) rewrites the template text and answers questions, given only tool outputs as context |
| 3. Hosted LLM | Never required | Not implemented by default; if added later, it must be opt-in and the app must work without it |

Guardrails for levels 2–3: the model receives structured tool results, is told to use only those numbers, and its output is checked so that every number in it appears in the tool results; otherwise we fall back to the template.

## Consequences

- The deployed app never breaks because of a missing key or quota.
- Interviewers see the full agent workflow and explanations online; you demonstrate the local LLM in a screen recording.
- Honest positioning: the "intelligence" lives in analytics, models and optimization; the LLM is a communication layer.
