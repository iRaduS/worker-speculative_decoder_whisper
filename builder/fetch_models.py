import os
from pathlib import Path
from faster_whisper.utils import download_model

model_names = [
    "large-v3",
    "distil-large-v3",
]

# Network volume path for RunPod serverless
NETWORK_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODELS_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "models")


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


if __name__ == "__main__":
    # Determine cache directory
    cache_dir = ensure_cache_directory()
    
    # Loop through models sequentially
    for model_name in model_names:
        download_model_weights(model_name, cache_dir)

    print("Finished processing all models.")
