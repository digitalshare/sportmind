"""SportMind AI — thin Streamlit demo client.

The laptop only uploads the video and renders results; ALL compute (ffmpeg, Whisper, YOLO)
runs in the serverless `video_worker` endpoint, and labeling runs on the Qwen2.5-7B vLLM
endpoint. Nothing heavy runs locally.

run:  streamlit run web/app.py
env:  RUNPOD_API_KEY (for the vLLM labeler)
      VIDEO_WORKER_URL  (default: local `flash dev` at http://localhost:8888/video_worker/runsync)
      VIDEO_WORKER_KEY  (bearer token when pointing at a deployed endpoint; omit for local dev)

NOTE: Flash scans every .py in the project and imports it to find @Endpoint functions, and
`flash deploy` treats an unimportable file as fatal. This dashboard depends on streamlit/requests
which are NOT in Flash's build env, so the entire app is guarded under `if __name__ == "__main__"`.
Streamlit executes the target script with __name__ == "__main__", so the UI still runs under
`streamlit run web/app.py`, while `import web.app` (Flash's scan) is a harmless no-op.
"""
import base64
import os
import sys
import tempfile

if __name__ == "__main__":
    import requests
    import streamlit as st

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from client import llm

    WORKER_URL = os.getenv("VIDEO_WORKER_URL", "http://localhost:8888/video_worker/runsync")
    WORKER_KEY = os.getenv("VIDEO_WORKER_KEY")

    st.set_page_config(page_title="SportMind AI", layout="wide")
    st.title("⚽ SportMind AI — Highlight Generator")
    st.caption("Upload a match clip → top highlights (audio-driven) + player/ball detection. "
               "All compute is serverless on RunPod.")

    top_n = st.sidebar.slider("Highlights", 3, 15, 8)
    uploaded = st.file_uploader("Match video", type=["mp4", "mov", "mkv", "webm"])

    def call_worker(video_bytes: bytes) -> dict:
        payload = {"input": {"input_data": {"video_b64": base64.b64encode(video_bytes).decode(),
                                            "top_n": top_n}}}
        headers = {"Authorization": f"Bearer {WORKER_KEY}"} if WORKER_KEY else {}
        r = requests.post(WORKER_URL, json=payload, headers=headers, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("output", data)  # RunPod wraps results in {"output": ...}

    if uploaded and st.button("Analyze", type="primary"):
        with st.spinner("Running serverless pipeline (cold start may take a minute)…"):
            result = call_worker(uploaded.getvalue())

        tab_h, tab_d = st.tabs(["Highlights", "Detection"])

        with tab_h:
            clips = result.get("clips", [])
            st.write(f"**{len(clips)} highlights** · duration {result.get('duration', 0):.0f}s")
            for c in clips:
                with st.spinner("Labeling…"):
                    tag = llm.label_highlight(c.get("transcript", ""), c.get("score", 0.0))
                cols = st.columns([1, 2])
                with cols[0]:
                    clip_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    with open(clip_path, "wb") as f:
                        f.write(base64.b64decode(c["clip_b64"]))
                    st.video(clip_path)
                with cols[1]:
                    st.markdown(f"### {tag['label']}  ·  `{c['t']:.0f}s`  ·  energy {c['score']:.2f}")
                    st.write(tag["summary"])
                    if c.get("transcript"):
                        st.caption(f"🎙️ {c['transcript']}")

        with tab_d:
            d = result.get("detections", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg players on screen", d.get("avg_players", 0))
            c2.metric("Ball presence", f"{100 * d.get('ball_presence', 0):.0f}%")
            c3.metric("Frames sampled", d.get("frames", 0))
            st.caption(f"YOLOv8n · sampled at {d.get('fps', 0)} fps (COCO person + sports ball)")
