#!/usr/bin/env python3
"""
Enhanced Coqui TTS Web Server
Comprehensive web interface for text-to-speech, voice cloning, and audio management
"""

import os
import sys
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import traceback

# Set environment variables before importing numba-dependent libraries
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'

# Create cache directory
cache_dir = os.environ.get('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.makedirs(cache_dir, exist_ok=True)

import torch
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import audio libraries with error handling
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Audio libraries not fully available: {e}")
    AUDIO_LIBS_AVAILABLE = False

# Add TTS to path
sys.path.insert(0, '/app')

# Import TTS with error handling
try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    TTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: TTS not fully available: {e}")
    TTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
class Config:
    PORT = int(os.getenv('PORT', 2201))
    MODELS_DIR = os.getenv('TTS_MODELS_DIR', '/app/models')
    VOICES_DIR = os.getenv('TTS_VOICES_DIR', '/app/voices')
    OUTPUTS_DIR = os.getenv('TTS_OUTPUTS_DIR', '/app/outputs')
    UPLOADS_DIR = os.getenv('TTS_UPLOADS_DIR', '/app/uploads')
    
    # Ensure directories exist
    for directory in [MODELS_DIR, VOICES_DIR, OUTPUTS_DIR, UPLOADS_DIR]:
        os.makedirs(directory, exist_ok=True)

config = Config()

# Global variables for model management
current_tts_model = None
model_manager = None
available_models = {}
voice_profiles = {}

def initialize_models():
    """Initialize TTS models and model manager"""
    global model_manager, available_models
    
    if not TTS_AVAILABLE:
        logger.error("TTS libraries not available, running in limited mode")
        available_models = {'tts_models': {}, 'vocoder_models': {}, 'voice_conversion_models': {}}
        return
    
    try:
        model_manager = ModelManager()
        
        # Get list of available models
        models_dict = model_manager.list_models()
        available_models = {
            'tts_models': models_dict.get('tts_models', {}),
            'vocoder_models': models_dict.get('vocoder_models', {}),
            'voice_conversion_models': models_dict.get('voice_conversion_models', {})
        }
        
        logger.info(f"Initialized with {len(available_models['tts_models'])} TTS models")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        available_models = {'tts_models': {}, 'vocoder_models': {}, 'voice_conversion_models': {}}

def load_voice_profiles():
    """Load existing voice profiles from disk"""
    global voice_profiles
    
    voices_file = os.path.join(config.VOICES_DIR, 'voice_profiles.json')
    try:
        if os.path.exists(voices_file):
            with open(voices_file, 'r') as f:
                voice_profiles = json.load(f)
        else:
            voice_profiles = {}
    except Exception as e:
        logger.error(f"Error loading voice profiles: {e}")
        voice_profiles = {}

def save_voice_profiles():
    """Save voice profiles to disk"""
    voices_file = os.path.join(config.VOICES_DIR, 'voice_profiles.json')
    try:
        with open(voices_file, 'w') as f:
            json.dump(voice_profiles, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving voice profiles: {e}")

def get_current_model():
    """Get or initialize the current TTS model"""
    global current_tts_model
    
    if not TTS_AVAILABLE:
        return None
    
    if current_tts_model is None:
        try:
            # Default to XTTS v2 for voice cloning capabilities
            device = "cuda" if torch.cuda.is_available() else "cpu"
            current_tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            logger.info(f"Loaded default XTTS v2 model on {device}")
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            # Fallback to simpler model
            try:
                current_tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                logger.info("Loaded fallback Tacotron2 model")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                current_tts_model = None
    
    return current_tts_model

# Routes
@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': current_tts_model is not None,
        'tts_available': TTS_AVAILABLE,
        'audio_libs_available': AUDIO_LIBS_AVAILABLE
    })

@app.route('/api/models')
def get_models():
    """Get available TTS models"""
    return jsonify({
        'available_models': available_models,
        'current_model': getattr(current_tts_model, 'model_name', None) if current_tts_model else None
    })

@app.route('/api/voices')
def get_voices():
    """Get saved voice profiles"""
    return jsonify(voice_profiles)

@app.route('/api/tts', methods=['POST'])
def synthesize_speech():
    """Main TTS synthesis endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice_profile = data.get('voice_profile')
        speaker_id = data.get('speaker_id')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        model = get_current_model()
        if not model:
            return jsonify({'error': 'No TTS model available'}), 500
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(config.OUTPUTS_DIR, filename)
        
        # Synthesize speech based on parameters
        if voice_profile and voice_profile in voice_profiles:
            # Use cloned voice
            speaker_wav = voice_profiles[voice_profile]['audio_path']
            model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path
            )
        elif speaker_id:
            # Use speaker ID for multi-speaker models
            model.tts_to_file(
                text=text,
                speaker=speaker_id,
                language=language,
                file_path=output_path
            )
        else:
            # Standard synthesis
            model.tts_to_file(
                text=text,
                file_path=output_path
            )
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/api/download/{filename}'
        })
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clone_voice', methods=['POST'])
def clone_voice():
    """Voice cloning endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        voice_name = request.form.get('voice_name', f'voice_{uuid.uuid4().hex[:8]}')
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded audio
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(config.VOICES_DIR, f"{voice_name}_{filename}")
        audio_file.save(audio_path)
        
        # Process audio for voice cloning (ensure proper format)
        duration = 0
        if AUDIO_LIBS_AVAILABLE:
            try:
                # Load and resample to 22050 Hz for better compatibility
                audio, sr = librosa.load(audio_path, sr=22050)
                sf.write(audio_path, audio, sr)
                duration = len(audio) / sr
            except Exception as e:
                logger.warning(f"Audio processing warning: {e}")
        else:
            logger.warning("Audio processing libraries not available, using original file")
        
        # Save voice profile
        voice_profiles[voice_name] = {
            'name': voice_name,
            'audio_path': audio_path,
            'created_at': datetime.now().isoformat(),
            'duration': duration
        }
        
        save_voice_profiles()
        
        return jsonify({
            'success': True,
            'voice_name': voice_name,
            'message': f'Voice "{voice_name}" cloned successfully'
        })
        
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated audio files"""
    try:
        return send_from_directory(config.OUTPUTS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/outputs')
def list_outputs():
    """List all generated audio files"""
    try:
        files = []
        for filename in os.listdir(config.OUTPUTS_DIR):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                filepath = os.path.join(config.OUTPUTS_DIR, filename)
                stat = os.stat(filepath)
                files.append({
                    'filename': filename,
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'download_url': f'/api/download/{filename}'
                })
        
        return jsonify({'files': sorted(files, key=lambda x: x['created'], reverse=True)})
    except Exception as e:
        logger.error(f"Error listing outputs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_voice/<voice_name>', methods=['DELETE'])
def delete_voice(voice_name):
    """Delete a voice profile"""
    try:
        if voice_name not in voice_profiles:
            return jsonify({'error': 'Voice profile not found'}), 404
        
        # Delete audio file
        audio_path = voice_profiles[voice_name]['audio_path']
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Remove from profiles
        del voice_profiles[voice_name]
        save_voice_profiles()
        
        return jsonify({'success': True, 'message': f'Voice "{voice_name}" deleted'})
        
    except Exception as e:
        logger.error(f"Error deleting voice: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/switch', methods=['POST'])
def switch_model():
    """Switch TTS model"""
    global current_tts_model
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'No model name provided'}), 400
        
        # Load new model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_tts_model = TTS(model_name).to(device)
        
        return jsonify({
            'success': True,
            'current_model': model_name,
            'device': device
        })
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice_conversion', methods=['POST'])
def voice_conversion():
    """Voice conversion endpoint"""
    try:
        if 'source_audio' not in request.files or 'target_voice' not in request.form:
            return jsonify({'error': 'Missing source audio or target voice'}), 400
        
        source_file = request.files['source_audio']
        target_voice = request.form.get('target_voice')
        
        if target_voice not in voice_profiles:
            return jsonify({'error': 'Target voice not found'}), 404
        
        # Save source audio temporarily
        source_path = os.path.join(config.UPLOADS_DIR, f"temp_{uuid.uuid4().hex[:8]}.wav")
        source_file.save(source_path)
        
        # Get target voice path
        target_path = voice_profiles[target_voice]['audio_path']
        
        # Generate output filename
        output_filename = f"converted_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(config.OUTPUTS_DIR, output_filename)
        
        # Perform voice conversion using XTTS
        model = get_current_model()
        if model and hasattr(model, 'voice_conversion'):
            model.voice_conversion(
                source_wav=source_path,
                target_wav=target_path,
                file_path=output_path
            )
        else:
            return jsonify({'error': 'Voice conversion not supported by current model'}), 400
        
        # Clean up temporary file
        os.remove(source_path)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        logger.error(f"Voice conversion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_synthesis', methods=['POST'])
def batch_synthesis():
    """Batch text-to-speech synthesis"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        voice_profile = data.get('voice_profile')
        language = data.get('language', 'en')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        model = get_current_model()
        if not model:
            return jsonify({'error': 'No TTS model available'}), 500
        
        results = []
        batch_id = uuid.uuid4().hex[:8]
        
        for i, text in enumerate(texts):
            filename = f"batch_{batch_id}_{i:03d}.wav"
            output_path = os.path.join(config.OUTPUTS_DIR, filename)
            
            # Synthesize based on voice profile
            if voice_profile and voice_profile in voice_profiles:
                speaker_wav = voice_profiles[voice_profile]['audio_path']
                model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=output_path
                )
            else:
                model.tts_to_file(
                    text=text,
                    file_path=output_path
                )
            
            results.append({
                'text': text,
                'filename': filename,
                'download_url': f'/api/download/{filename}'
            })
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def get_model_info():
    """Get detailed information about current model"""
    try:
        model = get_current_model()
        if not model:
            return jsonify({'error': 'No model loaded'}), 500
        
        info = {
            'model_name': getattr(model, 'model_name', 'Unknown'),
            'device': str(model.device) if hasattr(model, 'device') else 'Unknown',
            'language_manager': getattr(model, 'language_manager', None) is not None,
            'speaker_manager': getattr(model, 'speaker_manager', None) is not None,
            'supported_languages': [],
            'supported_speakers': []
        }
        
        # Get supported languages
        if hasattr(model, 'language_manager') and model.language_manager:
            info['supported_languages'] = list(model.language_manager.language_names)
        
        # Get supported speakers
        if hasattr(model, 'speaker_manager') and model.speaker_manager:
            info['supported_speakers'] = list(model.speaker_manager.speaker_names)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

# Static file serving
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# External API endpoints for container integration
@app.route('/api/v1/synthesize', methods=['POST'])
def api_synthesize():
    """External API endpoint for text synthesis"""
    try:
        # Support both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        text = data.get('text', '')
        voice_id = data.get('voice_id')
        language = data.get('language', 'en')
        output_format = data.get('format', 'wav')
        
        if not text:
            return jsonify({'error': 'Text parameter is required'}), 400
        
        model = get_current_model()
        if not model:
            return jsonify({'error': 'TTS service unavailable'}), 503
        
        # Generate filename
        filename = f"api_{uuid.uuid4().hex[:8]}.{output_format}"
        output_path = os.path.join(config.OUTPUTS_DIR, filename)
        
        # Synthesize
        if voice_id and voice_id in voice_profiles:
            speaker_wav = voice_profiles[voice_id]['audio_path']
            model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path
            )
        else:
            model.tts_to_file(
                text=text,
                file_path=output_path
            )
        
        return jsonify({
            'success': True,
            'audio_url': f'/api/download/{filename}',
            'filename': filename,
            'text': text,
            'voice_id': voice_id,
            'language': language
        })
        
    except Exception as e:
        logger.error(f"API synthesis error: {e}")
        return jsonify({'error': 'Synthesis failed', 'detail': str(e)}), 500

@app.route('/api/v1/voices', methods=['GET'])
def api_list_voices():
    """External API endpoint to list available voices"""
    try:
        voice_list = []
        for name, profile in voice_profiles.items():
            voice_list.append({
                'id': name,
                'name': name,
                'created_at': profile['created_at'],
                'duration': profile.get('duration', 0)
            })
        
        return jsonify({
            'voices': voice_list,
            'count': len(voice_list)
        })
    except Exception as e:
        logger.error(f"API list voices error: {e}")
        return jsonify({'error': 'Failed to list voices'}), 500

@app.route('/api/v1/clone', methods=['POST'])
def api_clone_voice():
    """External API endpoint for voice cloning"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file is required'}), 400
        
        audio_file = request.files['audio']
        voice_name = request.form.get('name', f'api_voice_{uuid.uuid4().hex[:8]}')
        
        # Save and process audio
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(config.VOICES_DIR, f"{voice_name}_{filename}")
        audio_file.save(audio_path)
        
        # Process audio
        duration = 0
        if AUDIO_LIBS_AVAILABLE:
            try:
                audio, sr = librosa.load(audio_path, sr=22050)
                sf.write(audio_path, audio, sr)
                duration = len(audio) / sr
            except Exception as e:
                logger.warning(f"Audio processing warning: {e}")
        
        # Save voice profile
        voice_profiles[voice_name] = {
            'name': voice_name,
            'audio_path': audio_path,
            'created_at': datetime.now().isoformat(),
            'duration': duration
        }
        
        save_voice_profiles()
        
        return jsonify({
            'success': True,
            'voice_id': voice_name,
            'message': f'Voice "{voice_name}" cloned successfully',
            'duration': duration
        })
        
    except Exception as e:
        logger.error(f"API voice cloning error: {e}")
        return jsonify({'error': 'Voice cloning failed', 'detail': str(e)}), 500

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """External API endpoint for service status"""
    try:
        return jsonify({
            'status': 'operational',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'cuda_available': torch.cuda.is_available(),
            'models_loaded': current_tts_model is not None,
            'tts_available': TTS_AVAILABLE,
            'audio_libs_available': AUDIO_LIBS_AVAILABLE,
            'voice_count': len(voice_profiles),
            'supported_formats': ['wav', 'mp3', 'flac'],
            'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko']
        })
    except Exception as e:
        logger.error(f"API status error: {e}")
        return jsonify({'error': 'Status check failed'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large'}), 413

if __name__ == '__main__':
    logger.info("Starting Enhanced Coqui TTS Web Server...")
    
    # Initialize models and voice profiles
    initialize_models()
    load_voice_profiles()
    
    logger.info(f"Server starting on port {config.PORT}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=False,
        threaded=True
    )