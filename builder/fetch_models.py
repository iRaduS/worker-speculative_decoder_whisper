import os
from pathlib import Path
from faster_whisper.utils import download_model

# Whisper models for faster-whisper
whisper_model_names = [
    "large-v3",
    "distil-large-v3",
]

# Transformers models for speculative decoding
transformers_models = [
    "openai/whisper-large-v3",
    "distil-whisper/distil-large-v3",
]

# Network volume path for RunPod serverless
NETWORK_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODELS_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "models")

# Set environment variables for transformers cache
if os.path.exists(NETWORK_VOLUME_PATH):
    os.environ["HF_HOME"] = MODELS_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = MODELS_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = MODELS_CACHE_DIR
    os.environ["TORCH_HOME"] = MODELS_CACHE_DIR


def check_model_exists(model_name, cache_dir):
    """
    Check if model already exists in cache directory.
    """
    if not cache_dir:
        return False
    
    model_path = Path(cache_dir) / model_name
    return model_path.exists() and any(model_path.iterdir())


def download_model_weights(selected_model, cache_dir=None):
    """
    Download model weights only if they don't already exist in cache.
    """
    if check_model_exists(selected_model, cache_dir):
        print(f"Model {selected_model} already exists in cache, skipping download.")
        return
    
    print(f"Downloading {selected_model} to {cache_dir or 'default cache'}...")
    download_model(selected_model, cache_dir=cache_dir)
    print(f"Finished downloading {selected_model}.")


def ensure_cache_directory():
    """
    Ensure the models cache directory exists.
    """
    if os.path.exists(NETWORK_VOLUME_PATH):
        os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
        print(f"Using network volume cache directory: {MODELS_CACHE_DIR}")
        return MODELS_CACHE_DIR
    else:
        print("Network volume not available, using default cache.")
        return None


def download_transformers_model(model_id, cache_dir=None):
    """
    Download transformers model if it doesn't exist in cache.
    """
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
        
        print(f"Checking/downloading transformers model: {model_id}")
        
        # Try to load from cache first
        if "whisper" in model_id.lower():
            AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False
            )
            AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        else:
            AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False
            )
        
        print(f"Transformers model {model_id} ready.")
    except Exception as e:
        print(f"Error downloading transformers model {model_id}: {e}")


if __name__ == "__main__":
    # Determine cache directory
    cache_dir = ensure_cache_directory()
    
    # Download faster-whisper models
    print("Downloading faster-whisper models...")
    for model_name in whisper_model_names:
        download_model_weights(model_name, cache_dir)
    
    # Download transformers models for speculative decoding
    print("Downloading transformers models...")
    for model_id in transformers_models:
        download_transformers_model(model_id, cache_dir)

    print("Finished processing all models.")
