"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import gc
import os
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
)  # Still needed for transcribe potentially?
import numpy as np
import time
import warnings

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp

# Speculative decoding imports
try:
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
    import torchaudio
    import soundfile as sf
    SPECULATIVE_AVAILABLE = True
except ImportError as e:
    SPECULATIVE_AVAILABLE = False
    print(f"Warning: Speculative decoding dependencies not available: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define available models (for validation)
AVAILABLE_MODELS = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
}

# Speculative decoding models
SPECULATIVE_MODELS = {
    "whisper-large-v3-speculative": {
        "main_model": "openai/whisper-large-v3",
        "assistant_model": "distil-whisper/distil-large-v3"
    },
}

TARGET_SR = 16_000  # Whisper target sample rate (Hz)


class Predictor:
    """A Predictor class for the Whisper model with lazy loading"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.models = {}
        self.model_lock = (
            threading.Lock()
        )  # Lock for thread-safe model loading/unloading

    def setup(self):
        """No models are pre-loaded. Setup is minimal."""
        pass

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",  # Added in a previous PR
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
    ):
        """
        Run a single prediction on the model, loading/unloading models as needed.
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models are: {AVAILABLE_MODELS}"
            )

        with self.model_lock:
            model = None
            if model_name not in self.models:
                # Unload existing model if necessary
                if self.models:
                    existing_model_name = list(self.models.keys())[0]
                    print(f"Unloading model: {existing_model_name}...")
                    # Remove reference and clear dict
                    del self.models[existing_model_name]
                    self.models.clear()
                    # Hint Python to release memory
                    gc.collect()
                    if rp_cuda.is_available():
                        # If using PyTorch models, you might call torch.cuda.empty_cache()
                        # FasterWhisper uses CTranslate2; explicit cache clearing might not be needed
                        # but gc.collect() is generally helpful.
                        pass
                    print(f"Model {existing_model_name} unloaded.")

                # Load the requested model
                print(f"Loading model: {model_name}...")
                try:
                    # Check for network volume cache directory
                    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
                    models_cache_dir = None
                    if os.path.exists(network_volume_path):
                        models_cache_dir = os.path.join(network_volume_path, "models")
                    
                    loaded_model = WhisperModel(
                        model_name,
                        device="cuda" if rp_cuda.is_available() else "cpu",
                        compute_type="float16" if rp_cuda.is_available() else "int8",
                        download_root=models_cache_dir
                    )
                    self.models[model_name] = loaded_model
                    model = loaded_model
                    print(f"Model {model_name} loaded successfully.")
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    raise ValueError(f"Failed to load model {model_name}: {e}") from e
            else:
                # Model already loaded
                model = self.models[model_name]
                print(f"Using already loaded model: {model_name}")

            # Ensure model is loaded before proceeding
            if model is None:
                raise RuntimeError(
                    f"Model {model_name} could not be loaded or retrieved."
                )

        # Model is now loaded and ready, proceed with prediction (outside the lock?)
        # Consider if transcribe is thread-safe or if it should also be within the lock
        # For now, keeping transcribe outside as it's CPU/GPU bound work

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # Note: FasterWhisper's transcribe might release the GIL, potentially allowing
        # other threads to acquire the model_lock if transcribe is lengthy.
        # If issues arise, the lock might need to encompass the transcribe call too.
        segments, info = list(
            model.transcribe(
                str(audio),
                language=language,
                task="transcribe",
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],  # Might need conversion from string
                without_timestamps=False,
                max_initial_timestamp=1.0,
                word_timestamps=word_timestamps,
                vad_filter=enable_vad,
            )
        )

        segments = list(segments)

        # Format transcription
        transcription_output = format_segments(transcription, segments)

        # Handle translation if requested
        translation_output = None
        if translate:
            translation_segments, _ = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature,  # Reuse temperature settings for translation
            )
            translation_output = format_segments(
                translation, list(translation_segments)
            )

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": transcription_output,
            "translation": translation_output,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if word_timestamps:
            word_timestamps_list = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps_list.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            results["word_timestamps"] = word_timestamps_list

        return results


class SpeculativePredictor:
    """A Predictor class for Whisper with speculative decoding"""

    def __init__(self):
        """Initializes the predictor with no models loaded."""
        self.main_model = None
        self.assistant_model = None
        self.processor = None
        self.device = None
        self.torch_dtype = None
        self.current_model_name = None
        self.model_lock = threading.Lock()

    def setup(self):
        """Initialize CUDA settings if available."""
        if not SPECULATIVE_AVAILABLE:
            raise RuntimeError("Speculative decoding dependencies not available. Please install required packages.")
        
        if torch.cuda.is_available() and not os.environ.get('FORCE_CPU'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Speculative decoding will use CUDA")
        else:
            print("Speculative decoding will use CPU")

    def _load_speculative_models(self, model_name):
        """Load the main and assistant models for speculative decoding."""
        if model_name not in SPECULATIVE_MODELS:
            raise ValueError(f"Invalid speculative model: {model_name}. Available: {list(SPECULATIVE_MODELS.keys())}")

        model_config = SPECULATIVE_MODELS[model_name]
        main_model_id = model_config["main_model"]
        assistant_model_id = model_config["assistant_model"]

        # Determine device and dtype
        force_cpu = os.environ.get('FORCE_CPU', False)
        if torch.cuda.is_available() and not force_cpu:
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            print("Warning: Running speculative decoding on CPU will be slower than CUDA.")

        print(f"Loading main model: {main_model_id}")
        start_time = time.time()
        
        # Check for network volume cache directory
        network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
        cache_dir = None
        if os.path.exists(network_volume_path):
            cache_dir = os.path.join(network_volume_path, "models")
            print(f"Using network volume cache for speculative models: {cache_dir}")
        
        # Load main model
        self.main_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            main_model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        self.main_model.to(self.device)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(main_model_id, cache_dir=cache_dir)

        # Load assistant model
        print(f"Loading assistant model: {assistant_model_id}")
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        )
        self.assistant_model.to(self.device)

        load_time = time.time() - start_time
        print(f"Models loaded in {load_time:.2f}s")
        self.current_model_name = model_name

    def _prepare_inputs(self, audio_path):
        """Prepare audio inputs for the model."""
        # Load and process audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            # Resample to 16kHz if needed
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
            audio_array = waveform.squeeze().numpy()
        except Exception:
            # Fallback to soundfile
            try:
                audio_array, sr = sf.read(audio_path)
                if sr != TARGET_SR:
                    print(f"Warning: Audio sample rate {sr} != {TARGET_SR}. Results may be suboptimal.")
            except Exception as e:
                raise RuntimeError(f"Could not load audio file {audio_path}: {e}")

        # Prepare inputs for the model
        inputs = self.processor(
            audio_array,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )
        
        # Move to device and convert dtype
        inputs = inputs.to(device=self.device, dtype=self.torch_dtype)
        
        # Ensure attention mask is properly set
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_features"])

        return inputs

    def _generate_with_speculative_decoding(self, inputs, language=None, task="transcribe", **kwargs):
        """Generate with speculative decoding."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()
        
        generate_kwargs = {
            "assistant_model": self.assistant_model,
            "task": task,
            "use_cache": True,
            "do_sample": False,
            "max_new_tokens": 128,
            **kwargs
        }
        
        if language:
            generate_kwargs["language"] = language

        outputs = self.main_model.generate(**inputs, **generate_kwargs)
        
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        generation_time = time.time() - start_time
        
        return outputs, generation_time

    def predict(
        self,
        audio,
        model_name="whisper-large-v3-speculative",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
    ):
        """Run speculative decoding inference."""
        with self.model_lock:
            # Load models if not already loaded or if different model requested
            if (self.main_model is None or 
                self.assistant_model is None or 
                self.processor is None or 
                self.current_model_name != model_name):
                
                # Clear existing models
                if self.main_model is not None:
                    del self.main_model
                    del self.assistant_model
                    del self.processor
                    gc.collect()
                    if torch.cuda.is_available() and not os.environ.get('FORCE_CPU'):
                        torch.cuda.empty_cache()
                
                self._load_speculative_models(model_name)

        # Prepare inputs
        inputs = self._prepare_inputs(audio)
        
        # Generate transcription
        outputs, gen_time = self._generate_with_speculative_decoding(
            inputs,
            language=language,
            task="transcribe",
            temperature=0.0 if temperature == 0 else temperature,
        )
        
        # Decode the output
        prediction = self.processor.batch_decode(outputs, skip_special_tokens=True, normalize=True)[0]
        
        # Create a simple segment object for compatibility with format_segments
        class SimpleSegment:
            def __init__(self, text):
                self.id = 0
                self.seek = 0
                self.start = 0.0
                self.end = 0.0
                self.text = text
                self.tokens = []
                self.temperature = temperature
                self.avg_logprob = 0.0
                self.compression_ratio = 0.0
                self.no_speech_prob = 0.0
        
        segments = [SimpleSegment(prediction)]
        
        # Format transcription using the existing format_segments function
        transcription_output = format_segments(transcription, segments)
        
        # Handle translation if requested
        translation_output = None
        if translate:
            trans_outputs, _ = self._generate_with_speculative_decoding(
                inputs,
                language=language,
                task="translate",
                temperature=0.0 if temperature == 0 else temperature,
            )
            trans_prediction = self.processor.batch_decode(trans_outputs, skip_special_tokens=True, normalize=True)[0]
            trans_segments = [SimpleSegment(trans_prediction)]
            translation_output = format_segments(translation, trans_segments)

        results = {
            "segments": serialize_segments(segments),
            "detected_language": language or "en",  # We don't have language detection in this implementation
            "transcription": transcription_output,
            "translation": translation_output,
            "device": self.device,
            "model": model_name,
            "inference_time": gen_time,
        }

        return results


def serialize_segments(transcript):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
            "id": segment.id,
            "seek": segment.seek,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "tokens": segment.tokens,
            "temperature": segment.temperature,
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
        }
        for segment in transcript
    ]


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":  # Added VTT case
        return write_vtt(segments)
    else:  # Default or unknown format
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        # Using the consistent timestamp format from previous PR
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
