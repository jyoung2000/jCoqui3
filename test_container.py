#!/usr/bin/env python3
"""
Test script for Coqui TTS Docker container
"""
import requests
import time
import sys
import json

BASE_URL = "http://localhost:2201"

def wait_for_service(timeout=60):
    """Wait for the service to be ready"""
    print("Waiting for service to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print("✅ Service is ready!")
                    return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    print("❌ Service failed to start within timeout")
    return False

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"CUDA Available: {data.get('cuda_available')}")
        print(f"Models Loaded: {data.get('models_loaded')}")
        print(f"TTS Available: {data.get('tts_available')}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_status_api():
    """Test status API endpoint"""
    print("\n🔍 Testing status API...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/status")
        data = response.json()
        print(f"API Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print(f"Voice Count: {data.get('voice_count')}")
        return True
    except Exception as e:
        print(f"❌ Status API failed: {e}")
        return False

def test_synthesis():
    """Test text-to-speech synthesis"""
    print("\n🔍 Testing TTS synthesis...")
    try:
        payload = {
            "text": "Hello, this is a test of the Coqui TTS system running in Docker!",
            "language": "en"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/synthesize",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Synthesis successful: {data.get('filename')}")
                print(f"Audio URL: {data.get('audio_url')}")
                return True
            else:
                print(f"❌ Synthesis failed: {data}")
                return False
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Synthesis test failed: {e}")
        return False

def test_voices_api():
    """Test voices listing API"""
    print("\n🔍 Testing voices API...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/voices")
        data = response.json()
        print(f"Available voices: {len(data.get('voices', []))}")
        return True
    except Exception as e:
        print(f"❌ Voices API failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🐸 Coqui TTS Container Test Suite")
    print("=" * 40)
    
    # Wait for service
    if not wait_for_service():
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health,
        test_status_api,
        test_voices_api,
        test_synthesis
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Container is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()