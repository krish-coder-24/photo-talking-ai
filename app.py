# app.py

import os
import gc
import sys
import time
import math
import shutil
import hashlib
import tempfile
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
from IPython.display import clear_output

from rich.console import Console
from rich.traceback import install as rich_traceback

# Add the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models.inference.moda_test import LiveVASAPipeline, emo_map, set_seed

# --- Config ---
set_seed(42)
DEFAULT_CFG_PATH = "configs/audio2motion/inference/inference.yaml"
DEFAULT_MOTION_MEAN_STD_PATH = "src/datasets/mean.pt"
DEFAULT_SILENT_AUDIO_PATH = "src/examples/silent-audio.wav"
OUTPUT_DIR = "gradio_output"
WEIGHTS_DIR = "pretrain_weights"
REPO_ID = "lixinyizju/moda"

console = Console()
rich_traceback(show_locals=False)

# --- Helpers ---
def cooldown_gpu(duration=7):
    start, width = time.time(), 20
    while time.time()-start < duration:
        t = time.time()-start
        fill = int((1 + __import__('math').sin(t*3)) * (width/2))
        bar = "#"*fill + "-"*(width-fill)
        sys.stdout.write(f"\r[ {bar} ] Cooling GPU... {duration-int(t)}s")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rGPU cooldowned ✅            \n")
    sys.stdout.flush()

def clean_gpu(threshold_gb=2):
    """Force GPU + Python garbage cleanup."""
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected.")
        gc.collect()
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    available = total - reserved

    print(f"Total GPU: {total:.2f} GB | Available: {available:.2f} GB")
    if available < threshold_gb:
        print("🧹 Clearing GPU cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        cooldown_gpu(6)
    else:
        print("✅ GPU has enough free memory, no cleanup needed.")
    gc.collect()

def download_weights():
    """Ensure model weights are available locally."""
    motion_model_file = os.path.join(WEIGHTS_DIR, "moda", "net-200.pth")
    if not os.path.exists(motion_model_file):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=WEIGHTS_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except GatedRepoError:
            raise gr.Error(f"Access to gated repo '{REPO_ID}'. Request access on HF.")
        except (RepositoryNotFoundError, RevisionNotFoundError):
            raise gr.Error(f"Repo '{REPO_ID}' not found.")
        except Exception as e:
            raise gr.Error(f"Failed to download models. Error: {e}")

def ensure_wav_format(audio_path):
    """Convert input audio to 16kHz mono WAV if needed."""
    if audio_path is None:
        return None
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == ".wav":
        return str(audio_path)

    try:
        audio = AudioSegment.from_file(audio_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
            audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        return wav_path
    except Exception as e:
        raise gr.Error(f"Failed to convert to WAV: {e}")

def file_sig(p):
    """Cheap file signature for hashing without full read."""
    try:
        st = os.stat(p)
        return f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}"
    except Exception:
        return os.path.basename(p)

def make_run_output_dir(source_image_path, audio_path, emotion_id, cfg_scale, user_id="anon"):
    key = f"{user_id}|{file_sig(source_image_path)}|{file_sig(audio_path)}|{emotion_id}|{cfg_scale}"
    run_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    run_dir = os.path.join(OUTPUT_DIR, f"run_{run_hash}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_hash

def process_audio_in_chunks(
    audio,
    run_output_dir,
    source_image_path,
    emotion_id,
    cfg_scale,
    progress=None,
    status_cb=None
):
    """
    Split audio into chunks, resume missing ones, concatenate, cleanup, return final path.
    """
    window_ms, stride_ms = 750, 750
    chunks = [audio[start:start + window_ms] for start in range(0, len(audio), stride_ms)]
    video_segments, temp_wavs = [], []
    total_chunks = len(chunks)
    start_time = time.time()

    def log(msg, kind="info"):
        print(msg)
        if status_cb:
            status_cb(msg, kind)

    try:
        # If final exists already, short-circuit in caller (we still keep this safety)
        final_path = os.path.join(run_output_dir, "final_video.mp4")
        if os.path.exists(final_path):
            log(f"⚡ Cached final exists: {final_path}")
            return final_path

        for idx, chunk in progress.tqdm(enumerate(chunks), total=total_chunks, desc="Generating Video"):
            if progress:
                progress(((idx) / total_chunks), desc=f"Chunk {idx+1}/{total_chunks}")

            wav_path = os.path.join(run_output_dir, f"chunk_{idx}.wav")
            mp4_path = os.path.join(run_output_dir, f"chunk_{idx}.mp4")

            # Resume: if mp4 already exists, just load and continue
            if os.path.exists(mp4_path):
                log(f"[Resume] Chunk {idx+1}/{total_chunks} already done.")
                try:
                    video_segments.append(VideoFileClip(mp4_path))
                except Exception as e:
                    raise RuntimeError(f"Failed to open existing video for chunk {idx}: {e}")
                continue

            log(f"==== Processing chunk ({idx+1}/{total_chunks}) ====")
            chunk.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            temp_wavs.append(wav_path)

            chunk_start = time.time()
            try:
                out_path = pipeline.driven_sample(
                    image_path=source_image_path,
                    audio_path=wav_path,
                    cfg_scale=float(cfg_scale),
                    emo=emotion_id,
                    save_dir=run_output_dir,
                    smooth=False,
                    silent_audio_path=DEFAULT_SILENT_AUDIO_PATH,
                )
                clean_gpu()

                if not out_path or not os.path.exists(out_path):
                    raise FileNotFoundError(f"Pipeline failed to produce video for chunk {idx}")

                # Normalize name for resume determinism
                if out_path != mp4_path:
                    try:
                        shutil.move(out_path, mp4_path)
                    except Exception:
                        # If move fails for any reason but the file exists, keep original path
                        mp4_path = out_path

                clear_output() if (idx % 10) == 0 else print()
            
            except RuntimeError as e:
                clean_gpu()
                raise gr.Error(
                    f"GPU OOM on chunk {idx}. Try shorter audio or lower settings.\nError: {e}"
                )

            # Load clip safely
            try:
                clip = VideoFileClip(mp4_path)
            except Exception as e:
                clean_gpu()
                raise RuntimeError(f"Failed to open video for chunk {idx}: {e}")

            video_segments.append(clip)

            # ETA
            elapsed = time.time() - start_time
            avg_per_chunk = elapsed / (idx + 1)
            remaining = max(0.0, avg_per_chunk * (total_chunks - idx - 1))
            log(f"Chunk {idx+1} done in {time.time()-chunk_start:.2f}s | ETA ~{remaining:.1f}s left")

            clean_gpu()

        if not video_segments:
            raise RuntimeError("No video segments were generated — pipeline produced nothing.")

        if progress:
            progress(0.98, desc="Concatenating video")

        final_clip = concatenate_videoclips(video_segments, method="compose")
        final_path = os.path.join(run_output_dir, "final_video.mp4")
        final_clip.write_videofile(final_path, codec="libx264", audio_codec="aac")

        # Cleanup intermediates (keep only final)
        for f in os.listdir(run_output_dir):
            if f.startswith("chunk_") or f.startswith("tmp_"):
                try:
                    os.remove(os.path.join(run_output_dir, f))
                except Exception:
                    pass

        log(f"✅ Final ready: {final_path}")
        return final_path

    finally:
        for w in temp_wavs:
            try:
                if os.path.exists(w):
                    os.remove(w)
            except Exception:
                pass
        for clip in video_segments:
            try:
                clip.close()
            except Exception:
                pass
        clean_gpu()

# --- Init ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
download_weights()

try:
    pipeline = LiveVASAPipeline(
        cfg_path=DEFAULT_CFG_PATH,
        motion_mean_std_path=DEFAULT_MOTION_MEAN_STD_PATH
    )
except Exception as e:
    print(f"Pipeline init failed: {e}")
    pipeline = None

emo_name_to_id = {v: k for k, v in emo_map.items()}

# --- Core ---
@spaces.GPU(duration=120)
def generate_motion(
    source_image_path,
    driving_audio_path,
    emotion_name,
    cfg_scale,
    existing_run_behavior,
    user_tag,
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
):
    if pipeline is None:
        raise gr.Error("Pipeline failed to initialize.")
    if not source_image_path:
        raise gr.Error("Upload a source image.")
    if not driving_audio_path:
        raise gr.Error("Upload a driving audio.")

    logs = []
    def status_cb(msg, kind="info"):
        logs.append(msg)

    def info(msg):
        print(msg); gr.Info(msg); logs.append(msg)

    def warn(msg):
        print(msg); gr.Warning(msg); logs.append(f"⚠️ {msg}")

    start_time = time.time()
    wav_audio_path = ensure_wav_format(driving_audio_path)
    temp_wav_created = wav_audio_path != driving_audio_path

    # user_id from request IP + user_tag
    ip = "anon"
    try:
        ip = getattr(request.client, "host", None) or request.headers.get("x-forwarded-for", "anon")
    except Exception:
        pass
    user_id = (user_tag.strip() or "user") + f"@{ip}"

    emotion_id = emo_name_to_id.get(emotion_name, 8)
    run_output_dir, run_hash = make_run_output_dir(source_image_path, wav_audio_path, emotion_id, cfg_scale, user_id=user_id)
    final_path = os.path.join(run_output_dir, "final_video.mp4")

    exists_final = os.path.exists(final_path)
    exists_chunks = False
    try:
        exists_chunks = any(f.startswith("chunk_") and f.endswith(".mp4") for f in os.listdir(run_output_dir))
    except Exception:
        exists_chunks = False

    # Decision based on radio
    info(f"Run ID: {run_hash}")
    if exists_final or exists_chunks:
        warn(f"Existing data found in {run_output_dir} — behavior: {existing_run_behavior}")

    if existing_run_behavior == "Use Cached Final" and exists_final:
        info(f"Using cached final: {final_path}")
        if temp_wav_created and os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)
        return final_path, "\n".join(logs)

    if existing_run_behavior == "Regenerate Fresh":
        # wipe directory
        for f in os.listdir(run_output_dir):
            try:
                os.remove(os.path.join(run_output_dir, f))
            except Exception:
                pass
        info("Cleared previous files. Starting fresh.")

    elif existing_run_behavior == "Resume":
        if exists_final:
            info(f"Final already exists; returning cached final: {final_path}")
            if temp_wav_created and os.path.exists(wav_audio_path):
                os.remove(wav_audio_path)
            return final_path, "\n".join(logs)
        else:
            info("Resuming from existing chunks (if any).")

    else:  # Auto
        if exists_final:
            info(f"Auto: returning cached final: {final_path}")
            if temp_wav_created and os.path.exists(wav_audio_path):
                os.remove(wav_audio_path)
            return final_path, "\n".join(logs)
        elif exists_chunks:
            info("Auto: partial chunks found → resuming.")
        else:
            info("Auto: no cache found → fresh generation.")

    # Process
    audio = AudioSegment.from_wav(wav_audio_path)
    try:
        result_path = process_audio_in_chunks(
            audio,
            run_output_dir,
            source_image_path,
            emotion_id,
            cfg_scale,
            progress=progress,
            status_cb=status_cb
        )
    finally:
        if temp_wav_created and os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)
        clean_gpu()

    gr.Info(f"Done in {time.time()-start_time:.2f}s → {result_path}")
    logs.append(f"Done in {time.time()-start_time:.2f}s → {result_path}")
    return result_path, logs[-1]

# --- Wrapper with ETA logging ---
def run_and_log(source_image_path, driving_audio_path, emotion_name, cfg_scale, existing_run_behavior, user_tag):
    if not driving_audio_path:
        return None, "⚠️ No audio provided!"

    try:
        # crude ETA guess
        audio = AudioSegment.from_wav(driving_audio_path)
        n_chunks = math.ceil(len(audio) / 500)
        eta_sec = n_chunks * 1.2  # heuristic
    except Exception as e:
        console.print_exception()  # pretty traceback to server logs
        return None, "❌ Failed to read audio file. Please upload a valid audio."

    start_time = time.time()
    try:
        final_path = generate_motion(
            source_image_path,
            driving_audio_path,
            emotion_name,
            cfg_scale,
            existing_run_behavior,
            user_tag
        )
    except RuntimeError as e:
        console.print_exception()
        return None, "❌ GPU ran out of memory. Try shorter audio or lower CFG scale."
    except Exception as e:
        console.print_exception()
        return None, f"❌ Generation failed: {str(e)}"

    elapsed = time.time() - start_time
    msg = (
        f"🎯 Estimated time: ~{eta_sec:.1f}s\n"
        f"⏱️ Actual time: {elapsed:.1f}s\n"
        f"✅ Generation completed successfully!"
    )
    return final_path, msg

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width:960px;margin:0 auto}") as demo:
    gr.HTML(
        """
        <div align='center'>
            <br>
            <h1>🎥 Project: Photo Talking AI</h1>
            <p><a href='https://github.com/krish-coder-24/photo-talking-ai/' target='_blank'>
            <img src='https://img.shields.io/badge/Code-Github-green'></a></p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("🚀 Generate Video"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 📥 Input Settings")
                    source_image = gr.Image(label="Source Image", type="filepath", value="src/examples/reference_images/7.jpg")
                    driving_audio = gr.Audio(label="Driving Audio", type="filepath", value="src/examples/driving_audios/5.wav")

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        emotion_dropdown = gr.Dropdown(
                            label="Emotion",
                            choices=list(emo_map.values()),
                            value="None"
                        )
                        cfg_slider = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0, maximum=3.0,
                            step=0.05, value=1.2
                        )
                        existing_run_behavior = gr.Radio(
                            label="If existing run detected",
                            choices=["Auto", "Resume", "Regenerate Fresh", "Use Cached Final"],
                            value="Auto",
                            interactive=True
                        )
                        user_tag = gr.Textbox(
                            label="User Tag (optional)",
                            placeholder="username / session / custom tag"
                        )

                    submit_button = gr.Button("🎬 Generate Video", variant="primary")

                with gr.Column(scale=1, min_width=450):
                    gr.Markdown("### 📺 Output")
                    output_video = gr.Video(label="Generated Video")

            # hidden initially
            status_box = gr.Textbox(
                label="Status / Logs",
                interactive=False,
                visible=False
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### 📌 About this Project  
                This is an academic demo for generating **talking photos** using AI.  
                - Upload a **source image**  
                - Provide a **driving audio**  
                - (Optional) Adjust emotion / CFG scale  

                ⚠️ **Disclaimer:**  
                Academic use only. You are liable for any generated content.  
                """
            )

    # Hook up with ETA-aware wrapper
    submit_button.click(
        fn=run_and_log,
        inputs=[source_image, driving_audio, emotion_dropdown, cfg_slider, existing_run_behavior, user_tag],
        outputs=[output_video, status_box]
    ).then(
        lambda: gr.update(visible=True),
        None,
        status_box
    )


if __name__ == "__main__":
    demo.launch(share=True)
