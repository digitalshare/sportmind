---
marp: true
theme: uncover
paginate: true
backgroundColor: #0d1117
color: #e6edf3
style: |
  section {
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 26px;
    text-align: left;
    justify-content: flex-start;
    padding: 60px 70px;
  }
  h1 { color: #58a6ff; font-size: 52px; }
  h2 { color: #58a6ff; font-size: 38px; }
  h3 { color: #7ee787; font-size: 28px; }
  strong { color: #ffa657; }
  a { color: #79c0ff; }
  table { font-size: 20px; }
  code { background: #161b22; color: #7ee787; }
  pre { background: #161b22; border-radius: 10px; font-size: 18px; }
  .lead { text-align: center; justify-content: center; }
  .pill { color: #8b949e; font-size: 20px; }
  footer { color: #484f58; font-size: 14px; }
---

<!-- _class: lead -->

# ⚽ SportMind AI

### An AI operating system for sports video

<br>

**Upload any match → tactical analysis, auto-highlights, predictions, and searchable insight.**

<span class="pill">Powered by RunPod Flash · serverless GPU microservices</span>

---

## The problem

Coaches, broadcasters, analysts, and creators **drown in raw footage**.

- A 90-minute match = hours of manual review for a few key moments.
- Existing AI tools are **single-purpose** — one app per task, per sport.
- Nothing composes vision + audio + text into one **tactical narrative**.

> What if every uploaded video became a queryable intelligence layer?

---

## The vision — one platform, many AI services

<div class="pill">Not a video analyzer. A modular AI OS for sports video.</div>

| | | |
|---|---|---|
| 🎬 **Highlights** | 🧠 **Tactical Coach** | 📈 **Live Prediction** |
| 📝 **Match Summary** | 🔎 **NL Video Search** | 💬 **Tactical Chat** |
| 🏷️ **Dataset Factory** | 🎧 **Multimodal Fusion** | 🚆 **Rail / Industrial** |

Each service is **independent**, **composable**, and **sport-agnostic** —
soccer today, basketball, tennis, esports, or railway inspection tomorrow.

---

## Why RunPod Flash

Each capability = an independent **serverless GPU `@Endpoint`**.

```python
@Endpoint(name="player_detect",
          gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
          dependencies=["ultralytics", "opencv-python-headless"])
async def detect_objects(frames): ...
```

- **Parallel** — spin up only the services a request needs.
- **Scale-to-zero** — no idle cost; RTX 4090 ≈ **$0.34/hr**.
- **No Dockerfiles** — `flash deploy`, one endpoint per function.

---

## Architecture — one pattern, every service

```
        Upload  →  Flash API Gateway
                        │
     ┌──────────────────┼──────────────────┐
   extract            analyze            compose
   (CPU/ffmpeg)   (GPU, parallel)      (CPU/score)
                        │
        ┌───────┬───────┼───────┬────────┐
      YOLO    Whisper   OCR   Pose AI  Vision-LLM
        └───────┴───────┼───────┴────────┘
                        │
              Knowledge Fusion → LLM narrative
                        │
        Highlights · Report · Search Index
```

**`extract → analyze → compose → narrate`** — reused by all 9 services.

---

## MVP shipped — 2 features, real architecture

### 🎬 1a · AI Highlight Generator
Audio energy + commentary → **top-10 labeled clips** + timeline.
*Mostly CPU; Whisper + LLM labels via Flash GPU + existing vLLM endpoint.*

### 🟦 1b · Player & Ball Detection
YOLO on RTX 4090 → **annotated overlay video** + on-screen stats.
*First GPU batch-inference endpoint — reused by tracking, pose, segmentation.*

<span class="pill">Both lay down the skeleton every later service plugs into.</span>

---

## Live demo flow

1. **Upload** `match.mp4` in the dashboard.
2. Highlights appear as the pipeline runs:

```
▶ 00:05:23   Goal
▶ 00:18:17   Bicycle kick
▶ 00:46:02   Great save
▶ 01:21:55   Winning goal
```

3. Switch tab → **annotated detection overlay** (players + ball).
4. *(Optional)* **Bright Data** scrapes the real match report to cross-label events.

---

## Roadmap

| Phase | Service | Adds |
|---|---|---|
| ✅ 1 | Highlights + Detection | the reusable pipeline |
| 2 | Match Summary | timeline → exec report (LLM) |
| 3 | Tactical Coach | tracking, formation, heatmaps |
| 4 | Search + Chat | frame embeddings, retrieval |
| 5 | Live Prediction | xG / win-probability |
| 6 | Dataset Factory · Trains | domain swap, same arch |

---

<!-- _class: lead -->

# SportMind AI

### Upload a match. Get intelligence.

**One Flash architecture · every sport · every modality.**

<span class="pill">Built on RunPod Flash serverless GPU · the ask: GPU credits + a pilot team</span>
