"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import os
import tempfile

# Set cache directories BEFORE importing any ML libraries
def setup_cache_directories():
    """Setup cache directories for network volume support."""
    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if os.path.exists(network_volume_path):
        models_cache_dir = os.path.join(network_volume_path, "models")
        os.makedirs(models_cache_dir, exist_ok=True)
        
        # Set environment variables for all caching libraries
        os.environ["HF_HOME"] = models_cache_dir
        os.environ["HF_HUB_CACHE"] = models_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = models_cache_dir
        os.environ["TORCH_HOME"] = models_cache_dir
        
        print(f"Using network volume cache: {models_cache_dir}")
        return models_cache_dir
    else:
        print("Network volume not available, using default cache.")
        return None

# Setup cache directories before any imports
setup_cache_directories()

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


# Initialize models on startup with network volume support
def initialize_models():
    """Initialize models with network volume caching."""
    print("Initializing models with network volume support...")
    
    # Run model fetching script to ensure models are cached
    try:
        import subprocess
        result = subprocess.run(["/usr/bin/python", "/fetch_models.py"], 
                              capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"Model fetching failed: {result.stderr}")
        else:
            print(f"Model fetching completed: {result.stdout}")
    except Exception as e:
        print(f"Error running fetch_models.py: {e}")
    
    # Set cache directory environment variable for faster-whisper
    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    if os.path.exists(network_volume_path):
        models_cache_dir = os.path.join(network_volume_path, "models")
        os.environ["HF_HUB_CACHE"] = models_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = models_cache_dir
        print(f"Using network volume cache: {models_cache_dir}")

# Initialize both predictors
initialize_models()

FASTER_WHISPER_MODEL = predict.Predictor()
FASTER_WHISPER_MODEL.setup()

SPECULATIVE_MODEL = None
try:
    SPECULATIVE_MODEL = predict.SpeculativePredictor()
    SPECULATIVE_MODEL.setup()
    print("Speculative decoding model initialized successfully")
except Exception as e:
    print(f"Speculative decoding model initialization failed: {e}")
    SPECULATIVE_MODEL = None


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    with rp_debugger.LineTimer('prediction_step'):
        # Choose the appropriate model based on inference method
        inference_method = job_input.get("inference_method", "faster_whisper")
        
        if inference_method == "speculative_decoding":
            if SPECULATIVE_MODEL is None:
                return {"error": "Speculative decoding is not available. Please check dependencies."}
            
            # For speculative decoding, we need to use the speculative model names
            model_name = job_input["model"]
            if model_name in predict.AVAILABLE_MODELS:
                # Convert regular model names to speculative equivalent
                model_name = "whisper-large-v3-speculative"
            
            whisper_results = SPECULATIVE_MODEL.predict(
                audio=audio_input,
                model_name=model_name,
                transcription=job_input["transcription"],
                translation=job_input["translation"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"]
            )
        else:
            # Use faster-whisper (default)
            whisper_results = FASTER_WHISPER_MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translation=job_input["translation"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"]
            )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
