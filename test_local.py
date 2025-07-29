#!/usr/bin/env python3
"""
Local testing script for the runpod worker
"""
import json
import os

from rp_handler import run_whisper_job

def test_faster_whisper():
    """Test faster-whisper inference"""
    print("=" * 50)
    print("Testing Faster-Whisper Inference")
    print("=" * 50)
    
    with open('test_input.json', 'r') as f:
        test_data = json.load(f)
    
    result = run_whisper_job(test_data)
    print("Result:", json.dumps(result, indent=2))
    return result

def test_speculative_decoding():
    """Test speculative decoding inference"""
    print("=" * 50)
    print("Testing Speculative Decoding Inference")
    print("=" * 50)
    
    try:
        with open('test_speculative_input.json', 'r') as f:
            test_data = json.load(f)
        
        result = run_whisper_job(test_data)
        print("Result:", json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Speculative decoding test failed: {e}")
        return None

if __name__ == "__main__":
    # Set CPU mode for local testing
    os.environ['FORCE_CPU'] = '1'
    
    print("Running local tests...")
    print(f"FORCE_CPU environment variable: {os.environ.get('FORCE_CPU')}")
    
    # Test faster-whisper
    fw_result = test_faster_whisper()
    
    # Test speculative decoding
    spec_result = test_speculative_decoding()
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Faster-Whisper: {'✓ PASSED' if fw_result and 'error' not in fw_result else '✗ FAILED'}")
    print(f"Speculative Decoding: {'✓ PASSED' if spec_result and 'error' not in spec_result else '✗ FAILED'}")