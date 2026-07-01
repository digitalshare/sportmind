"""Optional Bright Data enrichment — sourcing sample clips and scraping match reports.

Bright Data is OPTIONAL for Phase 1a/1b. In this project Bright Data is accessed through the
**Bright Data MCP** at build/dev time (the agent calls `mcp__brightdata__search_engine`,
`mcp__brightdata__scrape_as_markdown`, etc.), not via a runtime SDK. This module documents the
two intended uses and provides parsing helpers for whatever the MCP returns.

Use cases:
  1. Source a Creative-Commons sample match clip:
       mcp__brightdata__search_engine(query="creative commons soccer match full video")
       mcp__brightdata__scrape_as_markdown(url=<candidate page>)  -> find a downloadable CC clip
  2. Enrich highlight labels with real events:
       mcp__brightdata__scrape_as_markdown(url=<match report>)    -> teams, score, scorers+minutes
       then cross-label highlights whose timestamp is near a real event minute.
"""
import re


def parse_scorers(report_markdown: str) -> list[dict]:
    """Best-effort extraction of `Player 23'`-style scorer/minute pairs from a scraped report."""
    events = []
    for m in re.finditer(r"([A-Z][\w.\-]+(?:\s+[A-Z][\w.\-]+)?)\s*\(?(\d{1,3})['’]\)?", report_markdown):
        events.append({"player": m.group(1).strip(), "minute": int(m.group(2))})
    return events


def cross_label(timeline: list[dict], events: list[dict], tol_s: float = 30.0) -> list[dict]:
    """Tag highlights that fall within `tol_s` of a real scored-event minute."""
    out = []
    for t in timeline:
        near = [e for e in events if abs(e["minute"] * 60 - t["t"]) <= tol_s]
        out.append({**t, "real_event": near[0] if near else None})
    return out
