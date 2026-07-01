"""Thin client for the Qwen2.5-7B vLLM serverless endpoint.

This is *non-Flash* client code that invokes an already-deployed endpoint, so per the
project's AGENTS.md it talks to the endpoint directly (OpenAI-compatible API) rather than
going through the Flash lifecycle CLI. Endpoint lifecycle (create/scale/delete) is managed
out-of-band (RunPod MCP / dashboard); here we only call it.

Endpoint: id `zeemah9p7zoped`, model Qwen/Qwen2.5-7B-Instruct.
Base URL: https://api.runpod.ai/v2/<id>/openai/v1   (needs RUNPOD_API_KEY)
"""
import json
import os

VLLM_ENDPOINT_ID = os.getenv("VLLM_ENDPOINT_ID", "zeemah9p7zoped")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

LABELS = ["Goal", "Save", "Card", "Chance", "Other"]

_client = None


def _get_client():
    """Lazily build the OpenAI-compatible client (keeps this module importable without
    `openai`/RUNPOD_API_KEY present — e.g. when Flash scans the project for endpoints)."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=os.environ["RUNPOD_API_KEY"],
            base_url=f"https://api.runpod.ai/v2/{VLLM_ENDPOINT_ID}/openai/v1",
        )
    return _client


def label_highlight(transcript: str, energy_score: float) -> dict:
    """Classify one highlight and write a one-line summary from its transcript + crowd energy."""
    prompt = (
        "You are a soccer highlight tagger. Given a short clip's commentary transcript and a "
        f"normalized crowd-energy score (0..1), pick exactly one label from {LABELS} and write a "
        "one-sentence summary. Respond as compact JSON: "
        '{"label": "...", "summary": "..."}.\n\n'
        f"Energy: {energy_score:.2f}\nTranscript: {transcript or '(no speech detected)'}"
    )
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    text = resp.choices[0].message.content.strip()
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
        label = data.get("label", "Other")
        return {"label": label if label in LABELS else "Other", "summary": data.get("summary", "")}
    except Exception:
        return {"label": "Other", "summary": text[:200]}


def match_summary(timeline: list[dict], labels: list[dict]) -> str:
    """Compose a short match summary from the labeled highlight timeline (later Phase 2)."""
    lines = [f"- {t['t']}s [{l['label']}] {l['summary']}" for t, l in zip(timeline, labels)]
    prompt = "Write a 3-4 sentence match recap from these labeled highlights:\n" + "\n".join(lines)
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    # smoke test against the live endpoint (warms it on first call)
    print(label_highlight("And it's there! What a strike into the top corner!", 0.97))
