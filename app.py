# app.py

import os, sys, gc, time, math, shutil, hashlib, tempfile
from pathlib import Path

import torch, gradio as gr, spaces
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
from rich.console import Console
from rich.traceback import install as rich_traceback

# --- Local imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from models.inference.moda_test import LiveVASAPipeline, emo_map, set_seed

# --- Config ---
set_seed(42)
CFG_PATH = "configs/audio2motion/inference/inference.yaml"
MEAN_STD_PATH = "src/datasets/mean.pt"
SILENT_AUDIO = "src/examples/silent-audio.wav"
OUTPUT_DIR, WEIGHTS_DIR = "gradio_output", "pretrain_weights"
REPO_ID = "lixinyizju/moda"

console = Console()
rich_traceback(show_locals=False)

# ---------------- Helpers ---------------- #
def clean_gpu(threshold_gb=2):
    """Free GPU if memory low."""
    if not torch.cuda.is_available():
        gc.collect(); return
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total - reserved
    console.log(f"[GPU] Total={total:.1f}GB | Free={free:.1f}GB")
    if free < threshold_gb:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        gc.collect(); time.sleep(3)

def download_weights():
    path = os.path.join(WEIGHTS_DIR, "moda", "net-200.pth")
    if os.path.exists(path): return
    try:
        snapshot_download(REPO_ID, local_dir=WEIGHTS_DIR, local_dir_use_symlinks=False, resume_download=True)
    except GatedRepoError:
        raise gr.Error(f"Repo '{REPO_ID}' is gated. Request access on HF.")
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise gr.Error(f"Repo '{REPO_ID}' not found.")
    except Exception as e:
        console.print_exception(); raise gr.Error("Failed to fetch model weights.")

def ensure_wav(audio_path):
    """Return 16kHz mono wav path."""
    if not audio_path: return None
    p = Path(audio_path)
    if p.suffix.lower() == ".wav": return str(p)
    audio = AudioSegment.from_file(p)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav", parameters=["-ar", "16000", "-ac", "1"])
    return tmp.name

def file_sig(p):
    try:
        st = os.stat(p)
        return f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}"
    except Exception:
        return os.path.basename(p)

def run_dir(image, audio, emo, cfg, user="anon"):
    key = f"{user}|{file_sig(image)}|{file_sig(audio)}|{emo}|{cfg}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    d = os.path.join(OUTPUT_DIR, f"run_{h}")
    os.makedirs(d, exist_ok=True)
    return d, h

# ---------------- Chunk Processing ---------------- #
def process_chunks(audio, run_dir, img, emo, cfg, progress, log):
    window = 1000
    chunks = [audio[t:t+window] for t in range(0, len(audio), window)]
    videos, wavs = [], []
    final = os.path.join(run_dir, "final_video.mp4")

    if os.path.exists(final):
        log("⚡ Using cached final."); return final

    for i, chunk in progress.tqdm(enumerate(chunks), total=len(chunks), desc="Chunks"):
        mp4 = os.path.join(run_dir, f"chunk_{i}.mp4")
        if os.path.exists(mp4):
            log(f"[Resume] Chunk {i} ready."); videos.append(VideoFileClip(mp4)); continue

        wav = os.path.join(run_dir, f"chunk_{i}.wav")
        chunk.export(wav, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        wavs.append(wav)

        try:
            out = pipeline.driven_sample(
                image_path=img, audio_path=wav, cfg_scale=float(cfg),
                emo=emo, save_dir=run_dir, smooth=False, silent_audio_path=SILENT_AUDIO
            )
            if not out or not os.path.exists(out): raise RuntimeError("Pipeline failed.")
            if out != mp4: shutil.move(out, mp4)
            videos.append(VideoFileClip(mp4))
        except Exception:
            console.print_exception(); raise gr.Error(f"❌ Failed on chunk {i}.")
        finally:
            clean_gpu()

    if not videos: raise gr.Error("❌ No video segments generated.")

    final_clip = concatenate_videoclips(videos, method="compose")
    final_clip.write_videofile(final, codec="libx264", audio_codec="aac")

    for w in wavs: os.remove(w) if os.path.exists(w) else None
    for v in videos: v.close()
    return final

# ---------------- Core ---------------- #
@spaces.GPU(duration=120)
def generate_motion(img, audio, emo_name, cfg, behavior, user, progress=gr.Progress(track_tqdm=True), request: gr.Request=None):
    if not pipeline: raise gr.Error("Pipeline init failed.")
    if not img or not audio: raise gr.Error("Missing inputs.")

    wav = ensure_wav(audio)
    emo = emo_name_to_id.get(emo_name, 8)
    uid = (user.strip() or "user") + f"@{getattr(request.client,'host','anon')}"
    run, h = run_dir(img, wav, emo, cfg, uid)
    final = os.path.join(run, "final_video.mp4")

    exists_final = os.path.exists(final)
    exists_chunks = any(f.startswith("chunk_") for f in os.listdir(run))

    console.log(f"[Run {h}] Behavior={behavior}")

    if behavior == "Use Cached Final" and exists_final: return final, "⚡ Used cached final."
    if behavior == "Regenerate Fresh":
        [os.remove(os.path.join(run,f)) for f in os.listdir(run)]
    elif behavior == "Resume" and exists_final: return final, "⚡ Already complete."
    elif behavior == "Auto":
        if exists_final: return final, "⚡ Auto: cached final."
        if exists_chunks: console.log("Resuming…")

    audio_seg = AudioSegment.from_wav(wav)
    out = process_chunks(audio_seg, run, img, emo, cfg, progress, console.log)
    return out, f"✅ Done in {time.time():.1f}s"

# ---------------- Wrapper ---------------- #
def run_and_log(img, audio, emo, cfg, behavior, user):
    if not audio: return None, "⚠️ No audio."
    try:
        return generate_motion(img, audio, emo, cfg, behavior, user)
    except Exception:
        console.print_exception(); return None, "❌ Generation failed."

# ---------------- UI ---------------- #
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container{max-width:960px}") as demo:
    gr.HTML("<h1 align='center'>🎥 Photo Talking AI</h1>")

    with gr.Tab("🚀 Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                source_image = gr.Image(label="Image", type="filepath", value="src/examples/reference_images/7.jpg")
                driving_audio = gr.Audio(label="Audio", type="filepath", value="src/examples/driving_audios/5.wav")
                emotion = gr.Dropdown(label="Emotion", choices=list(emo_map.values()), value="None")
                cfg = gr.Slider(1.0, 3.0, 1.2, 0.05, label="CFG")
                behavior = gr.Radio(["Auto","Resume","Regenerate Fresh","Use Cached Final"], value="Auto")
                user_tag = gr.Textbox(label="User Tag")
                submit = gr.Button("🎬 Generate", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Result")
                status = gr.Textbox(label="Status", visible=False)

    submit.click(fn=run_and_log,
                 inputs=[source_image, driving_audio, emotion, cfg, behavior, user_tag],
                 outputs=[output_video, status]).then(lambda: gr.update(visible=True), None, status)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True); download_weights()
    try:
        pipeline = LiveVASAPipeline(cfg_path=CFG_PATH, motion_mean_std_path=MEAN_STD_PATH)
    except Exception: console.print_exception(); pipeline=None
    emo_name_to_id = {v:k for k,v in emo_map.items()}
    demo.launch(share=True)
