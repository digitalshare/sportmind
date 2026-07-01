# SportMind AI — Phased Build Plan (serverless-first MVP)

## Context

`ideas.md` proposes **SportMind AI**, an AI sports-video intelligence platform built as a
collection of independent RunPod serverless GPU microservices (9 services: YOLO, RTMPose,
SAM2, ByteTrack, Whisper, PaddleOCR, embeddings, tactical LLM, etc.).

That full vision is a multi-month build. The goal here is to ship **1–2 easy, high-impact
features first** on a **serverless-first** architecture that the demo runs in the cloud
(not on the laptop) and that every later service plugs into.

**Why serverless-first (and a note on speed):** serverless is *not* automatically faster.
GPU steps must run remotely anyway (the dev Mac has no NVIDIA GPU). Moving lightweight CPU glue
(ffmpeg) to a *separate* serverless CPU endpoint would actually be slower (network round-trips,
cold starts, Flash CPU endpoints pinned to **EU-RO-1**). The win comes from a **hosted, laptop-
independent, sharable** demo — and the way to keep it fast is to **upload the video once** and
do everything in-datacenter, so bytes don't shuttle local↔cloud. So ffmpeg stays serverless but
**co-located inside the GPU worker**, not as its own endpoint.

**Current state (verified via RunPod MCP):**
- Greenfield repo — `ideas.md` + this plan + the `deck/`. No app code yet, not a git repo.
- No RocketRide in this project (the `.claude/rules/rocketride.md` rule is stale here; ignore it).
- RunPod: **RTX 4090 (24 GB)** in stock — community **$0.34/hr**, secure $0.69/hr (pool `ADA_24`).
- **vLLM serverless endpoint (re)deployed** for highlight labeling/summary:
  - id **`zeemah9p7zoped`**, name `sportmind-qwen25-7b`
  - model **`Qwen/Qwen2.5-7B-Instruct`** (replaces the deleted tiny `qwen/qwen2.5-0.5b`)
  - image `runpod/worker-v1-vllm:v2.22.5`, RTX 4090, `min=0` scale-to-zero, FlashBoot on,
    `MAX_MODEL_LEN=8192`, `GPU_MEMORY_UTILIZATION=0.95`
  - OpenAI-compatible base URL: `https://api.runpod.ai/v2/zeemah9p7zoped/openai/v1`

**Decisions locked with the user:**
- First features: **(1a) audio-driven Highlight Generator, then (1b) Player/Ball Detection**.
- Architecture: **one serverless GPU "video worker"** (video uploaded once; ffmpeg + Whisper +
  YOLO run co-located in-datacenter; only small results returned). Local side is a thin client.
- Compute: **RunPod Flash SDK** (`@Endpoint` decorators, `flash deploy`) for the video worker;
  **vLLM serverless** for the LLM.
- Tooling: operate through the **RunPod MCP** and **Bright Data MCP**.
- Demo surface: **simple web dashboard** (Streamlit).

## Architecture

```
 Streamlit (local, thin client)
   │  upload video ONCE (base64 / presigned URL)
   ▼
 Flash @Endpoint  process_video   ── RTX 4090, scale-to-zero ──┐  in-datacenter, no shuttling:
   • ffmpeg: extract mono audio + sample frames                │   ffmpeg + Whisper + YOLO
   • RMS energy envelope → peak pick → top-N highlight windows │   co-located on one worker
   • Whisper: transcribe top-N windows                         │
   • YOLO: detect person / sports-ball on sampled frames       │
   • ffmpeg: cut top-N ±8s clips                               │
   returns ONLY: timeline + top-N clips(base64) + transcripts + detection stats
   ▼
 vLLM @ RunPod  zeemah9p7zoped (Qwen2.5-7B-Instruct)  ←── label/summarize each highlight
   ▼
 Streamlit renders: clips + labels + annotated detection preview
```

- **One worker, not many.** Avoids N cold starts + N video transfers. Detection can graduate to
  its own `@Endpoint` later (Phase 1b scaling) without changing the client contract.
- **Returning clips:** MVP returns top-N ±8 s clips as **base64** in the JSON response
  (top-10 short clips ≈ a few MB). If size grows, write clips to a RunPod **network volume** /
  object storage and return URLs.
- **Bright Data:** optional / on demand — (a) source a CC sample clip, (b) scrape a match report
  to enrich highlight labels. Skip unless a step needs web data.

## Project layout

```
sportmind/
├── flash_app/
│   ├── video_worker.py      # @Endpoint process_video: ffmpeg + Whisper + YOLO (RTX 4090)
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env.example          # RUNPOD_API_KEY, VLLM_ENDPOINT_ID=zeemah9p7zoped
├── client/
│   ├── llm.py                # vLLM client (OpenAI-compatible) for labels/summary
│   └── brightdata.py         # optional: source CC clip / scrape match report (BD MCP)
├── web/app.py                # Streamlit: upload → call process_video → render results
├── deck/                     # Marp pitch deck (update arch slide to single-worker design)
├── samples/                  # short test clip(s)
└── README.md
```

## Step 1 — Bigger model (DONE — verify before demo)
The Qwen2.5-7B vLLM endpoint `zeemah9p7zoped` is already provisioned via the RunPod MCP.
- `client/llm.py`: OpenAI-compatible client → `https://api.runpod.ai/v2/zeemah9p7zoped/openai/v1`,
  or call via `mcp__runpod__runsync-endpoint` (input `{messages:[...], sampling_params:{...}}`).
- 7B weights (~15 GB) → first cold start is slow; **warm before demoing** (one dummy request) or
  briefly set `min=1`. Verify with `mcp__runpod__endpoint-health` / `mcp__runpod__get-billing`.

## Phase 1a — AI Highlight Generator (first shippable feature)
`flash_app/video_worker.py::process_video`
`@Endpoint(gpu=GpuType.NVIDIA_GEFORCE_RTX_4090, dependencies=["openai-whisper","ultralytics","ffmpeg-python","opencv-python-headless","numpy"])`:
1. Decode uploaded video (base64 → temp file).
2. ffmpeg → mono audio; numpy short-window **RMS energy envelope**; pick sustained-loudness
   peaks (crowd roar + commentator spikes); take **top-N**.
3. ffmpeg-cut top-N ±8 s clips; Whisper-transcribe each clip's audio.
4. Feed transcript + energy to the vLLM endpoint to label each highlight
   (`Goal`/`Save`/`Card`/`Chance`) + one-line summary.
5. Return `{ timeline:[{t,score}], clips:[base64], transcripts:[...] }`.
- **(Optional Bright Data):** scrape the real match report to cross-label highlights with
  actual events/minutes.
- **Deliverable:** upload match → top-10 labeled highlight clips + timeline.

## Phase 1b — Player & Ball Detection (compute-heavy GPU path)
Add YOLO to the same worker (or split to its own `@Endpoint` if it needs independent scaling):
1. Sample frames (e.g. 2 fps), run pretrained YOLO (COCO `person` + `sports ball`), draw boxes.
2. Return annotated preview MP4 (base64) + stats (players on screen over time, ball-presence track).
- **Deliverable:** annotated overlay video + on-screen counts. Establishes the GPU batch-inference
  path that tracking/pose/segmentation reuse later.

## Web dashboard (spans 1a + 1b)
`web/app.py` (Streamlit): `st.file_uploader` → call `process_video` → list clips with `st.video`
+ labels → tab for the annotated detection preview. The laptop only uploads + renders — all
compute is hosted. This is the live demo surface.

## Later phases (roadmap, not built now)
- **2 — Match Summary** (timeline + Qwen2.5-7B → exec summary/report).
- **3 — Tactical Coach** (ByteTrack tracking → team ID → formation/heatmaps/passing network →
  tactical LLM; the 7B model gives real reasoning headroom now).
- **4 — NL Video Search & Chat** (SigLIP/DINOv2 frame embeddings + vector index).
- **5 — Prediction** (continuous xG/win-prob; Bright Data for real rosters/xG benchmarks).
- **6 — Dataset Factory / OCR / domain-swap to trains** (architecture identical, models swap).

## Verification (end-to-end)
1. **Model up:** `mcp__runpod__get-endpoint zeemah9p7zoped` ready; `mcp__runpod__runsync-endpoint`
   with a chat prompt returns sensible text (confirms Qwen2.5-7B is live, not the old 0.5B).
2. **Worker local:** `flash dev` → `curl localhost:8888/video_worker/runsync -d '{"input":{"input_data":{"video_b64":"..."}}}'`
   on a short `samples/` clip → returns timeline + clips + detections; check `localhost:8888/docs`.
3. **Pipeline:** peaks land on loud moments; top-N clips are watchable; YOLO boxes hit players/ball.
4. **LLM labels:** highlights get sensible labels/summaries from the 7B endpoint.
5. **Dashboard:** `streamlit run web/app.py`, upload sample, verify clips + labels + detection render
   — confirm nothing but the upload runs locally.
6. **Deploy + hosted demo:** `flash deploy`; point Streamlit at the deployed URL; re-run
   upload→result fully in the cloud. Watch cost via `mcp__runpod__get-billing` (scope `serverless`).

## Demo slides (pitch deck) — BUILT
Marp markdown at `deck/sportmind.md` → `deck/sportmind.html` / `deck/sportmind.pdf`
(`npx @marp-team/marp-cli deck/sportmind.md -o deck/sportmind.html`).
*Update the architecture slide to the single-worker, video-uploaded-once design and the 7B model.*

## Cost / risk notes
- Keep workers `min=0` (scale-to-zero); expect cold starts — FlashBoot / a warm-up call mitigates.
- **7B cold start** (~15 GB weights) is the main new latency source — warm before the live demo.
- Flash needs Python 3.10+ (host is darwin — fine).

## Sources
- [RunPod Flash overview](https://docs.runpod.io/flash/overview) ·
  [worker-vllm](https://github.com/runpod-workers/worker-vllm) ·
  [Deploy vLLM on Serverless](https://docs.runpod.io/serverless/vllm/get-started) ·
  [Serverless overview](https://docs.runpod.io/serverless/overview)
