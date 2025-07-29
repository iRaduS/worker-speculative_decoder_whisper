#!/bin/bash

echo "🚀 Starting RunPod Whisper Worker Local Test"
echo "============================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build the CPU image
echo "🔨 Building CPU Docker image with verbose logging..."
docker build --progress=plain --no-cache -f Dockerfile.cpu -t whisper-worker-cpu .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Run tests
echo "🧪 Running tests..."
docker run --rm whisper-worker-cpu python test_local.py

if [ $? -eq 0 ]; then
    echo "✅ Tests completed successfully!"
    echo ""
    echo "🎉 Your RunPod Whisper Worker is ready!"
    echo ""
    echo "To run the worker interactively:"
    echo "  docker run --rm -it -p 8000:8000 whisper-worker-cpu"
    echo ""
    echo "To test with curl:"
    echo "  curl -X POST http://localhost:8000 \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d @test_input.json"
else
    echo "❌ Tests failed"
    exit 1
fi