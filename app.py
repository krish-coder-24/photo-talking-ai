# app.py

import os
import gc
import sys
import time
import math
import gradio as gr
import spaces
import torch
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips

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

# --- Helpers ---
def clean_gpu(threshold_gb=2):
    """Force GPU + Python garbage cleanup."""
    if not torch.cuda.is_available():
        print("⚠️ No GPU detected.")
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
        cooldown_gpu(8)
    else:
        print("✅ GPU has enough free memory, no cleanup needed.")
    gc.collect()

def cooldown_gpu(duration=7):
    start, width = time.time(), 20
    while time.time()-start < duration:
        # sine wave pulse between 0 and width
        t = time.time()-start
        fill = int((1 + __import__('math').sin(t*3)) * (width/2))
        bar = "#"*fill + "-"*(width-fill)
        sys.stdout.write(f"\r[ {bar} ] Cooling GPU... {duration-int(t)}s")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\rGPU cooldowned ✅            \n")
    sys.stdout.flush()

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

def process_audio_in_chunks(audio, run_output_dir, source_image_path, emotion_id, cfg_scale):
    """Split audio into chunks, generate video segments, and return concatenated video path."""
    window_ms, stride_ms = 500, 500  # exact 1s chunks, no overlap
    chunks = [audio[start:start + window_ms] for start in range(0, len(audio), stride_ms)]
    video_segments, temp_files = [], []

    try:
        for idx, chunk in enumerate(chunks):
            print(f"==== Processing chunk ({idx+1}/{len(chunks)}) ====")
            tmp_chunk_path = os.path.join(run_output_dir, f"chunk_{idx}.wav")
            chunk.export(tmp_chunk_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            temp_files.append(tmp_chunk_path)

            try:
                out_path = pipeline.driven_sample(
                    image_path=source_image_path,
                    audio_path=tmp_chunk_path,
                    cfg_scale=float(cfg_scale),
                    emo=emotion_id,
                    save_dir=run_output_dir,
                    smooth=False,
                    silent_audio_path=DEFAULT_SILENT_AUDIO_PATH,
                )
                clean_gpu()

                # ✅ Ensure pipeline returned a file
                if not out_path or not os.path.exists(out_path):
                    raise FileNotFoundError(f"Pipeline failed to produce video for chunk {idx}")

            except RuntimeError as e:
                clean_gpu()
                raise gr.Error(
                    f"GPU OOM on chunk {idx}. "
                    f"Try shorter audio or lower settings.\nError: {e}"
                )

            # Load clip safely
            try:
                clip = VideoFileClip(out_path)
            except Exception as e:
                clean_gpu()
                raise RuntimeError(f"Failed to open video for chunk {idx}: {e}")

            video_segments.append(clip)
            clean_gpu()

        # ✅ Final concatenation
        if not video_segments:
            raise RuntimeError("No video segments were generated — pipeline produced nothing.")

        final_clip = concatenate_videoclips(video_segments, method="compose")
        final_path = os.path.join(run_output_dir, "final_video.mp4")
        final_clip.write_videofile(final_path, codec="libx264", audio_codec="aac")
        return final_path

    finally:
        # Cleanup temp files + GPU + video handles
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        for clip in video_segments:
            clip.close()
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
def generate_motion(source_image_path, driving_audio_path, emotion_name, cfg_scale, progress=gr.Progress(track_tqdm=True)):
    if pipeline is None:
        raise gr.Error("Pipeline failed to initialize.")
    if not source_image_path:
        raise gr.Error("Upload a source image.")
    if not driving_audio_path:
        raise gr.Error("Upload a driving audio.")

    start_time = time.time()
    wav_audio_path = ensure_wav_format(driving_audio_path)
    temp_wav_created = wav_audio_path != driving_audio_path

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    emotion_id = emo_name_to_id.get(emotion_name, 8)
    audio = AudioSegment.from_wav(wav_audio_path)

    try:
        final_path = process_audio_in_chunks(audio, run_output_dir, source_image_path, emotion_id, cfg_scale)
    finally:
        if temp_wav_created and os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)
        clean_gpu()

    print(f"Done in {time.time()-start_time:.2f}s → {final_path}")
    return final_path

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width:960px;margin:0 auto}") as demo:
    gr.HTML(
        """
        <div align='center'>
            <h1>Project: Photo talking AI</h1>
            <p>
                <a href='https://github.com/krish-coder-24/photo-talking-ai/'><img src='https://img.shields.io/badge/Code-Github-green'></a>
            </p>
        </div>
        """
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            source_image = gr.Image(label="Source Image", type="filepath", value="src/examples/reference_images/7.jpg")
            driving_audio = gr.Audio(label="Driving Audio", type="filepath", value="src/examples/driving_audios/5.wav")
            emotion_dropdown = gr.Dropdown(label="Emotion", choices=list(emo_map.values()), value="None")
            cfg_slider = gr.Slider(label="CFG Scale", minimum=1.0, maximum=3.0, step=0.05, value=1.2)
            submit_button = gr.Button("Generate Video", variant="primary")
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    gr.Markdown("---\n### Disclaimer\nAcademic use only. Users are liable for generated content.")

    submit_button.click(fn=generate_motion, inputs=[source_image, driving_audio, emotion_dropdown, cfg_slider], outputs=output_video)

if __name__ == "__main__":
    demo.launch(share=True)
