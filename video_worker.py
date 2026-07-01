# SportMind AI — serverless video worker (Phase 1a + 1b).
#
# One GPU @Endpoint that receives a video ONCE and does everything in-datacenter:
#   ffmpeg (audio + clips) -> RMS peak-pick -> Whisper transcribe -> YOLO detect.
# Returns only small artifacts (timeline + base64 clips + transcripts + detection stats);
# the raw video never shuttles back. Labeling/summary is done by the client against the
# Qwen2.5-7B vLLM endpoint (kept separate — see client/llm.py).
#
# run locally:  flash dev    (serves at localhost:8888, /docs explorer)
# deploy:       flash deploy
from runpod_flash import Endpoint, GpuType


@Endpoint(
    name="video_worker",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),  # scale-to-zero; cold start on first burst
    dependencies=["faster-whisper", "ultralytics", "opencv-python-headless", "numpy"],
    system_dependencies=["ffmpeg", "libgl1"],
)
async def process_video(input_data: dict) -> dict:
    """Audio-driven highlight detection + player/ball detection on an uploaded video.

    input_data:
      video_b64 (str, required) : base64-encoded video file
      top_n     (int)           : number of highlights to return (default 10)
      pad_s     (float)         : seconds of context each side of a peak (default 8.0)
      detect_fps(float)         : frame sample rate for YOLO detection (default 2.0)

    returns: { duration, timeline:[{t,score}], clips:[{t,score,clip_b64,transcript}],
               detections:{fps, frames, avg_players, ball_presence} }
    """
    import base64
    import os
    import subprocess
    import tempfile
    import wave

    import numpy as np

    top_n = int(input_data.get("top_n", 10))
    pad_s = float(input_data.get("pad_s", 8.0))
    detect_fps = float(input_data.get("detect_fps", 2.0))

    workdir = tempfile.mkdtemp(prefix="sportmind_")
    video_path = os.path.join(workdir, "input.mp4")
    with open(video_path, "wb") as f:
        f.write(base64.b64decode(input_data["video_b64"]))

    def _run(cmd):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- 1. extract mono 16 kHz audio --------------------------------------
    wav_path = os.path.join(workdir, "audio.wav")
    _run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", wav_path])

    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        audio = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio) / sr if sr else 0.0

    # --- 2. RMS energy envelope (0.5 s windows) ----------------------------
    win = max(1, int(0.5 * sr))
    n_win = len(audio) // win
    if n_win == 0:
        return {"duration": duration, "timeline": [], "clips": [], "detections": {}}
    frames = audio[: n_win * win].reshape(n_win, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-9)
    # normalize to 0..1 for an interpretable score
    rms_norm = (rms - rms.min()) / (rms.ptp() + 1e-9)
    times = (np.arange(n_win) + 0.5) * (win / sr)

    # --- 3. peak-pick: local maxima, >=10 s apart, take top-N --------------
    min_gap_win = max(1, int(10.0 / (win / sr)))
    order = np.argsort(rms_norm)[::-1]
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= min_gap_win for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= top_n:
            break
    chosen.sort()
    timeline = [{"t": round(float(times[i]), 2), "score": round(float(rms_norm[i]), 3)} for i in chosen]

    # --- 4. cut clips + Whisper transcribe each ----------------------------
    from faster_whisper import WhisperModel

    asr = WhisperModel("base", device="cuda", compute_type="float16")
    clips = []
    for peak in timeline:
        t = peak["t"]
        start = max(0.0, t - pad_s)
        dur = min(duration, t + pad_s) - start
        clip_path = os.path.join(workdir, f"clip_{int(t)}.mp4")
        _run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{dur:.2f}",
              "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", clip_path])
        # transcribe the clip's audio
        clip_wav = os.path.join(workdir, f"clip_{int(t)}.wav")
        _run(["ffmpeg", "-y", "-i", clip_path, "-ac", "1", "-ar", "16000", "-vn", clip_wav])
        segments, _ = asr.transcribe(clip_wav)
        text = " ".join(seg.text for seg in segments).strip()
        with open(clip_path, "rb") as cf:
            clip_b64 = base64.b64encode(cf.read()).decode()
        clips.append({"t": t, "score": peak["score"], "clip_b64": clip_b64, "transcript": text})

    # --- 5. YOLO person / sports-ball detection over sampled frames --------
    import cv2
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")  # COCO: class 0 = person, 32 = sports ball
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / detect_fps)))
    player_counts, ball_hits, sampled = [], 0, 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            res = yolo(frame, classes=[0, 32], verbose=False)[0]
            cls = res.boxes.cls.tolist() if res.boxes is not None else []
            player_counts.append(sum(1 for c in cls if int(c) == 0))
            if any(int(c) == 32 for c in cls):
                ball_hits += 1
            sampled += 1
        fi += 1
    cap.release()
    detections = {
        "fps": detect_fps,
        "frames": sampled,
        "avg_players": round(float(np.mean(player_counts)), 2) if player_counts else 0.0,
        "ball_presence": round(ball_hits / sampled, 3) if sampled else 0.0,
    }

    return {"duration": round(duration, 2), "timeline": timeline, "clips": clips, "detections": detections}


if __name__ == "__main__":
    import asyncio
    import base64
    import sys

    # smoke test: python video_worker.py path/to/clip.mp4
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/sample.mp4"
    with open(path, "rb") as f:
        payload = {"video_b64": base64.b64encode(f.read()).decode(), "top_n": 5}
    out = asyncio.run(process_video(payload))
    print({k: (v if k != "clips" else f"{len(v)} clips") for k, v in out.items()})
