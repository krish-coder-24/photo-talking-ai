# app.py

import os
import sys
import time
import gradio as gr
import spaces
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from pathlib import Path
import tempfile
from pydub import AudioSegment

# Add the src directory to the system path to allow for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.inference.moda_test import LiveVASAPipeline, emo_map, set_seed

# --- Configuration ---
# Set seed for reproducibility
set_seed(42)

# Paths and constants for the Gradio demo
DEFAULT_CFG_PATH = "configs/audio2motion/inference/inference.yaml"
DEFAULT_MOTION_MEAN_STD_PATH = "src/datasets/mean.pt"
DEFAULT_SILENT_AUDIO_PATH = "src/examples/silent-audio.wav"
OUTPUT_DIR = "gradio_output"
WEIGHTS_DIR = "pretrain_weights"
REPO_ID = "lixinyizju/moda"

# --- Download Pre-trained Weights from Hugging Face Hub ---
def download_weights():
    """
    Downloads pre-trained weights from Hugging Face Hub if they don't exist locally.
    """
    # A simple check for a key file to see if the download is likely complete
    motion_model_file = os.path.join(WEIGHTS_DIR, "moda", "net-200.pth")
    
    if not os.path.exists(motion_model_file):
        print(f"Weights not found locally. Downloading from Hugging Face Hub repo '{REPO_ID}'...")
        print(f"This may take a while depending on your internet connection.")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=WEIGHTS_DIR,
                local_dir_use_symlinks=False,  # Use False to copy files directly; safer for Windows
                resume_download=True,
            )
            print("Weights downloaded successfully.")
        except GatedRepoError:
            raise gr.Error(f"Access to the repository '{REPO_ID}' is gated. Please visit https://huggingface.co/{REPO_ID} to request access.")
        except (RepositoryNotFoundError, RevisionNotFoundError):
            raise gr.Error(f"The repository '{REPO_ID}' was not found. Please check the repository ID.")
        except Exception as e:
            print(f"An error occurred during download: {e}")
            raise gr.Error(f"Failed to download models. Please check your internet connection and try again. Error: {e}")
    else:
        print(f"Found existing weights at '{WEIGHTS_DIR}'. Skipping download.")

# --- Audio Conversion Function ---
def ensure_wav_format(audio_path):
    """
    Ensures the audio file is in WAV format. If not, converts it to WAV.
    Returns the path to the WAV file (either original or converted).
    """
    if audio_path is None:
        return None
    
    audio_path = Path(audio_path)
    
    # Check if already WAV
    if audio_path.suffix.lower() == '.wav':
        print(f"Audio is already in WAV format: {audio_path}")
        return str(audio_path)
    
    # Convert to WAV
    print(f"Converting audio from {audio_path.suffix} to WAV format...")
    
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wav_path = tmp_file.name
            # Export as WAV with standard settings
            audio.export(
                wav_path,
                format='wav',
                parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono - adjust if your model needs different settings
            )
        
        print(f"Audio converted successfully to: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"Error converting audio: {e}")
        raise gr.Error(f"Failed to convert audio file to WAV format. Error: {e}")

# --- Initialization ---
# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download weights before initializing the pipeline
download_weights()

# Instantiate the pipeline once to avoid reloading models on every request
print("Initializing MoDA pipeline...")
try:
    pipeline = LiveVASAPipeline(
        cfg_path=DEFAULT_CFG_PATH,
        motion_mean_std_path=DEFAULT_MOTION_MEAN_STD_PATH
    )
    print("MoDA pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing pipeline: {e}")
    pipeline = None

# Invert the emo_map for easy lookup from the dropdown value
emo_name_to_id = {v: k for k, v in emo_map.items()}

# --- Core Generation Function ---
@spaces.GPU(duration=120)
def generate_motion(source_image_path, driving_audio_path, emotion_name, cfg_scale, progress=gr.Progress(track_tqdm=True)):
    """
    The main function that takes Gradio inputs and generates the talking head video.
    """
    if pipeline is None:
        raise gr.Error("Pipeline failed to initialize. Check the console logs for details.")
        
    if source_image_path is None:
        raise gr.Error("Please upload a source image.")
    if driving_audio_path is None:
        raise gr.Error("Please upload a driving audio file.")

    start_time = time.time()
    
    # Ensure audio is in WAV format
    wav_audio_path = ensure_wav_format(driving_audio_path)
    temp_wav_created = wav_audio_path != driving_audio_path
    
    # Create a unique subdirectory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # Get emotion ID from its name
    emotion_id = emo_name_to_id.get(emotion_name, 8)  # Default to 'None' (ID 8) if not found

    print(f"Starting generation with the following parameters:")
    print(f"  Source Image: {source_image_path}")
    print(f"  Driving Audio (original): {driving_audio_path}")
    print(f"  Driving Audio (WAV): {wav_audio_path}")
    print(f"  Emotion: {emotion_name} (ID: {emotion_id})")
    print(f"  CFG Scale: {cfg_scale}")

    try:
        # Call the pipeline's inference method with the WAV audio
        result_video_path = pipeline.driven_sample(
            image_path=source_image_path,
            audio_path=wav_audio_path,
            cfg_scale=float(cfg_scale),
            emo=emotion_id,
            save_dir=".",
            smooth=False, # Smoothing can be slow, disable for a faster demo
            silent_audio_path=DEFAULT_SILENT_AUDIO_PATH,
        )
    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An unexpected error occurred: {str(e)}. Please check the console for details.")
    finally:
        # Clean up temporary WAV file if created
        if temp_wav_created and os.path.exists(wav_audio_path):
            try:
                os.remove(wav_audio_path)
                print(f"Cleaned up temporary WAV file: {wav_audio_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {wav_audio_path}: {e}")

    end_time = time.time()
    
    processing_time = end_time - start_time
    
    result_video_path = Path(result_video_path)
    final_path = result_video_path.with_name(f"final_{result_video_path.stem}{result_video_path.suffix}")
    
    print(f"Video generated successfully at: {final_path}")
    print(f"Processing time: {processing_time:.2f} seconds.")

    return final_path

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 960px !important; margin: 0 auto !important}") as demo:
    gr.HTML(
        """
        <div align='center'>
            <h1>MoDA: Multi-modal Diffusion Architecture for Talking Head Generation</h1>
            <p style="display:flex">
                <a href='https://lixinyyang.github.io/MoDA.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
                <a href='https://arxiv.org/abs/2507.03256'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                <a href='https://github.com/lixinyyang/MoDA/'><img src='https://img.shields.io/badge/Code-Github-green'></a>
            </p>
        </div>
        """
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            with gr.Row():
                source_image = gr.Image(label="Source Image", type="filepath", value="src/examples/reference_images/7.jpg")
            
            with gr.Row():
                driving_audio = gr.Audio(
                    label="Driving Audio",
                    type="filepath",
                    value="src/examples/driving_audios/5.wav"
                )

            with gr.Row():
                emotion_dropdown = gr.Dropdown(
                    label="Emotion",
                    choices=list(emo_map.values()),
                    value="None"
                )

            with gr.Row():
                cfg_slider = gr.Slider(
                    label="CFG Scale",
                    minimum=1.0,
                    maximum=3.0,
                    step=0.05,
                    value=1.2
                )
            
            submit_button = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    gr.Markdown(
        """
        ---
        ### **Disclaimer**
        This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using this generative model.
        """
    )
    
    submit_button.click(
        fn=generate_motion,
        inputs=[source_image, driving_audio, emotion_dropdown, cfg_slider],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(share=True)