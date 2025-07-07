#!/usr/bin/env python3
"""
Quick test script to verify the model loading functionality
without building the full Docker container
"""

import sys
import os
sys.path.insert(0, '/mnt/d/ClaudeCodeProjects/jCoqui3Backup')

try:
    from web_server.app import initialize_models, available_models, coqui_speakers, model_capabilities
    print("✅ Successfully imported app components")
    
    # Test model initialization
    print("🔄 Testing model initialization...")
    initialize_models()
    
    print(f"📊 Available models categories: {list(available_models.keys())}")
    
    if 'tts_models' in available_models:
        tts_count = 0
        for lang, datasets in available_models['tts_models'].items():
            for dataset, models in datasets.items():
                tts_count += len(models)
        print(f"🎯 Total TTS models: {tts_count}")
        
        # Show sample models
        print("📝 Sample TTS models:")
        count = 0
        for lang, datasets in available_models['tts_models'].items():
            for dataset, models in datasets.items():
                for model_name in models.keys():
                    print(f"  - tts_models/{lang}/{dataset}/{model_name}")
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break
            if count >= 5:
                break
    
    print(f"🎤 Coqui speakers loaded: {len(coqui_speakers)}")
    if coqui_speakers:
        print("📝 Sample Coqui speakers:")
        for i, (name, info) in enumerate(list(coqui_speakers.items())[:3]):
            print(f"  - {name}: {info.get('description', 'No description')}")
    
    print(f"⚙️ Model capabilities loaded: {len(model_capabilities)}")
    print(f"📝 Available capabilities: {list(model_capabilities.keys())}")
    
    print("\n✅ Model loading test completed successfully!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()