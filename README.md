![Faster Whisper Logo](https://5ccaof7hvfzuzf4p.public.blob.vercel-storage.com/banner-pjbGKw0buxbWGhMVC165Gf9qgqWo7I.jpeg)

[Faster Whisper](https://github.com/guillaumekln/faster-whisper) is designed to process audio files using various Whisper models, with options for transcription formatting, language translation and more.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-faster_whisper)](https://www.runpod.io/console/hub/runpod-workers/worker-faster_whisper)

---

## Models

### Faster-Whisper Models (Default)
- tiny
- base
- small
- medium
- large-v1
- large-v2
- large-v3
- distil-large-v2
- distil-large-v3
- turbo

### Speculative Decoding Models
- whisper-large-v3-speculative (2x faster inference with identical quality)

## Input

| Input                               | Type  | Description                                                                                                                                                            |
| ----------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audio`                             | Path  | URL to Audio file                                                                                                                                                      |
| `audio_base64`                      | str   | Base64-encoded audio file                                                                                                                                              |
| `model`                             | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "distil-large-v2", "distil-large-v3", "turbo", "whisper-large-v3-speculative". Default: "base" |
| `inference_method`                  | str   | Choose inference method. Choices: "faster_whisper", "speculative_decoding". Default: "faster_whisper"                                                                |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                  |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                                                                                         |
| `translation`                       | str   | Choose the format for the translation. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                                    |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                                                                                |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                                                                            |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                               |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                                                                                   |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: None                                                                                                         |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: None                                                                                                       |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                                                                           |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                                                                               |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True                                                                       |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                                                                                        |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4                                                                    |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0                                                                   |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6                                                               |
| `enable_vad`                        | bool  | If True, use the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model. Default: False               |
| `word_timestamps`                   | bool  | If True, include word timestamps in the output. Default: False                                                                                                         |

### Examples

#### Faster-Whisper (Default)
```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "turbo"
  }
}
```

#### Speculative Decoding (2x Faster)
```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "whisper-large-v3-speculative",
    "inference_method": "speculative_decoding",
    "language": "en"
  }
}
```

producing an output like this:

```json
{
  "segments": [
    {
      "id": 1,
      "seek": 106,
      "start": 0.11,
      "end": 3.11,
      "text": " Hello and welcome!",
      "tokens": [50364, 25, 7, 287, 50514],
      "temperature": 0.1,
      "avg_logprob": -0.8348079785480325,
      "compression_ratio": 0.5789473684210527,
      "no_speech_prob": 0.1453857421875
    }
  ],
  "detected_language": "en",
  "transcription": "Hello and welcome!",
  "translation": null,
  "device": "cuda",
  "model": "turbo",
  "translation_time": 0.3796223163604736
}
```

## Speculative Decoding

This worker now supports **Speculative Decoding** for faster Whisper inference with identical output quality. Speculative decoding achieves approximately **2x speedup** by using a smaller assistant model (distil-whisper) to predict tokens, which are then verified by the main model.

### Features
- **2x faster inference** while maintaining identical transcription quality
- **GPU optimized** for CUDA environments
- **Automatic fallback** to faster-whisper if speculative decoding is unavailable
- **Compatible API** with existing faster-whisper parameters

### Usage
To use speculative decoding, set the inference method in your request:

```json
{
  "input": {
    "audio": "your_audio_url_here",
    "inference_method": "speculative_decoding",
    "model": "whisper-large-v3-speculative",
    "language": "en"
  }
}
```

### Requirements
- CUDA-compatible GPU
- Additional dependencies: torch, transformers, torchaudio, soundfile, pydub

### Performance Benefits
- **Speed**: ~2x faster transcription compared to standard faster-whisper
- **Quality**: Identical transcription results (same WER scores)
- **Memory**: Efficient GPU memory usage with FP16 precision
- **Throughput**: Higher files-per-minute processing rate

## Local Development & Testing

### CPU Testing (No GPU Required)

For local development and testing without CUDA/GPU:

```bash
# Build CPU-only image
docker build -f Dockerfile.cpu -t whisper-worker-cpu .

# Run CPU tests
docker run --rm -it whisper-worker-cpu python test_local.py

# Run interactive container
docker run --rm -it -p 8000:8000 whisper-worker-cpu
```

### CUDA/GPU Production

For production deployment with GPU acceleration:

```bash
# Build CUDA image
docker build -f Dockerfile.cuda -t whisper-worker-cuda .

# Run with GPU support
docker run --gpus all --rm -it whisper-worker-cuda
```

### Local Development Without Docker

```bash
# Install CPU dependencies
pip install -r builder/requirements-cpu.txt

# Set CPU mode
export FORCE_CPU=1

# Run local tests
python test_local.py

# Or run the handler directly
python src/rp_handler.py
```

### Available Dockerfiles

- `Dockerfile` - Default CPU version for local development
- `Dockerfile.cpu` - CPU-only version for testing without GPU
- `Dockerfile.cuda` - GPU-optimized version for production deployment
