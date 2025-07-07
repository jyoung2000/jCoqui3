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
loaded_models = {}  # Cache for loaded models
coqui_speakers = {}  # Pre-trained Coqui speakers
model_capabilities = {}  # Track what each model can do

def initialize_models():
    """Initialize TTS models and model manager with enhanced capabilities"""
    global model_manager, available_models, coqui_speakers, model_capabilities
    
    # Always initialize capabilities and speakers first
    _initialize_model_capabilities()
    _load_coqui_speakers()
    
    # Always use fallback models to ensure immediate availability
    logger.info("Using comprehensive fallback model system for immediate availability...")
    available_models = _get_fallback_models()
    
    if not TTS_AVAILABLE:
        logger.warning("TTS libraries not available, using fallback models only")
        logger.info(f"Fallback mode: Loaded {_count_nested_models(available_models['tts_models'])} TTS models")
        return
    
    # Try to initialize ModelManager but don't let it block startup
    try:
        logger.info("Attempting to initialize TTS ModelManager in background...")
        import threading
        
        def _background_model_init():
            global model_manager
            try:
                model_manager = ModelManager()
                logger.info("ModelManager initialized successfully in background")
            except Exception as e:
                logger.warning(f"ModelManager initialization failed, continuing with fallback: {e}")
        
        # Start background initialization but don't wait for it
        init_thread = threading.Thread(target=_background_model_init, daemon=True)
        init_thread.start()
        
        logger.info("Background model initialization started, continuing with fallback models...")
        
    except Exception as e:
        logger.warning(f"Error in background model initialization: {e}")
    
    # Count total models
    total_tts_models = _count_nested_models(available_models['tts_models'])
    total_vocoder_models = _count_nested_models(available_models['vocoder_models'])
    
    logger.info(f"✅ Initialized with {total_tts_models} TTS models")
    logger.info(f"✅ Initialized with {total_vocoder_models} vocoder models")
    logger.info(f"✅ Available Coqui speakers: {len(coqui_speakers)}")
    
    # Debug: Log some example models
    if available_models['tts_models']:
        logger.info("📝 Sample TTS models available:")
        count = 0
        for lang in list(available_models['tts_models'].keys())[:3]:
            for dataset in list(available_models['tts_models'][lang].keys())[:2]:
                for model in list(available_models['tts_models'][lang][dataset].keys())[:2]:
                    logger.info(f"  - tts_models/{lang}/{dataset}/{model}")
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break
            if count >= 5:
                break

def _count_nested_models(models_dict):
    """Count total models in nested dictionary structure"""
    count = 0
    if isinstance(models_dict, dict):
        for lang_models in models_dict.values():
            if isinstance(lang_models, dict):
                for dataset_models in lang_models.values():
                    if isinstance(dataset_models, dict):
                        count += len(dataset_models)
    return count

def _get_fallback_models():
    """Provide a comprehensive fallback list of known TTS models"""
    return {
        'tts_models': {
            'multilingual': {
                'multi-dataset': {
                    'xtts_v2': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'XTTS v2 - Advanced voice cloning and multilingual synthesis',
                        'capabilities': model_capabilities.get('xtts_v2', {}),
                        'full_name': 'tts_models/multilingual/multi-dataset/xtts_v2'
                    },
                    'bark': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Bark - High-quality multilingual TTS with emotions',
                        'capabilities': model_capabilities.get('bark', {}),
                        'full_name': 'tts_models/multilingual/multi-dataset/bark'
                    },
                    'your_tts': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'YourTTS - Cross-lingual voice cloning',
                        'capabilities': model_capabilities.get('your_tts', {}),
                        'full_name': 'tts_models/multilingual/multi-dataset/your_tts'
                    }
                }
            },
            'en': {
                'ljspeech': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Tacotron2 with Dynamic Decoder Constraint',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/en/ljspeech/tacotron2-DDC'
                    },
                    'glow-tts': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Glow-TTS - Fast and stable TTS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/en/ljspeech/glow-tts'
                    },
                    'speedy-speech': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Speedy Speech - Very fast TTS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/en/ljspeech/speedy-speech'
                    }
                },
                'multi-dataset': {
                    'tortoise-v2': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Tortoise v2 - High-quality voice cloning',
                        'capabilities': model_capabilities.get('tortoise', {}),
                        'full_name': 'tts_models/en/multi-dataset/tortoise-v2'
                    }
                },
                'vctk': {
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'VITS multi-speaker model',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/en/vctk/vits'
                    }
                }
            },
            'es': {
                'mai': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Spanish Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/es/mai/tacotron2-DDC'
                    }
                }
            },
            'fr': {
                'mai': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'French Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/fr/mai/tacotron2-DDC'
                    }
                }
            },
            'de': {
                'thorsten': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'German Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/de/thorsten/tacotron2-DDC'
                    },
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'German VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/de/thorsten/vits'
                    }
                },
                'css10': {
                    'vits-neon': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'German CSS10 VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/de/css10/vits-neon'
                    }
                }
            },
            'it': {
                'mai': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Italian Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/it/mai/tacotron2-DDC'
                    }
                }
            },
            'pt': {
                'cv': {
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Portuguese VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/pt/cv/vits'
                    }
                }
            },
            'pl': {
                'mai': {
                    'glow-tts': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Polish Glow-TTS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/pl/mai/glow-tts'
                    }
                }
            },
            'tr': {
                'common-voice': {
                    'glow-tts': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Turkish Glow-TTS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/tr/common-voice/glow-tts'
                    }
                }
            },
            'ru': {
                'ruslan': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Russian Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/ru/ruslan/tacotron2-DDC'
                    }
                }
            },
            'nl': {
                'mai': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Dutch Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/nl/mai/tacotron2-DDC'
                    }
                }
            },
            'cs': {
                'cv': {
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Czech VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/cs/cv/vits'
                    }
                }
            },
            'ar': {
                'cv': {
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Arabic VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/ar/cv/vits'
                    }
                }
            },
            'zh': {
                'baker': {
                    'tacotron2-DDC-GST': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Chinese Tacotron2 with GST',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/zh/baker/tacotron2-DDC-GST'
                    }
                }
            },
            'ja': {
                'kokoro': {
                    'tacotron2-DDC': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Japanese Tacotron2',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/ja/kokoro/tacotron2-DDC'
                    }
                }
            },
            'hu': {
                'css10': {
                    'vits': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Hungarian VITS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/hu/css10/vits'
                    }
                }
            },
            'ko': {
                'kss': {
                    'glow-tts': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Korean Glow-TTS',
                        'capabilities': model_capabilities.get('standard', {}),
                        'full_name': 'tts_models/ko/kss/glow-tts'
                    }
                }
            }
        },
        'vocoder_models': {
            'universal': {
                'libri-tts': {
                    'fullband-melgan': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'Universal Fullband MelGAN',
                        'full_name': 'vocoder_models/universal/libri-tts/fullband-melgan'
                    }
                }
            },
            'en': {
                'ljspeech': {
                    'hifigan_v2': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'HiFiGAN v2',
                        'full_name': 'vocoder_models/en/ljspeech/hifigan_v2'
                    }
                }
            }
        },
        'voice_conversion_models': {
            'multilingual': {
                'vctk': {
                    'freevc24': {
                        'model_file': None,
                        'config_file': None,
                        'description': 'FreeVC24 voice conversion',
                        'full_name': 'voice_conversion_models/multilingual/vctk/freevc24'
                    }
                }
            }
        }
    }

def _initialize_model_capabilities():
    """Initialize capabilities for different model types"""
    global model_capabilities
    
    model_capabilities = {
        'xtts_v2': {
            'supports_emotions': True,
            'supports_voice_cloning': True,
            'supports_multilingual': True,
            'supports_speakers': True,
            'emotion_controls': ['temperature', 'repetition_penalty', 'length_penalty'],
            'languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko']
        },
        'bark': {
            'supports_emotions': True,
            'supports_voice_cloning': True,
            'supports_multilingual': True,
            'supports_speakers': True,
            'emotion_controls': ['temperature', 'voice_dirs'],
            'languages': ['en', 'de', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'tr', 'zh']
        },
        'tortoise': {
            'supports_emotions': True,
            'supports_voice_cloning': True,
            'supports_multilingual': False,
            'supports_speakers': True,
            'emotion_controls': ['preset', 'num_autoregressive_samples', 'diffusion_iterations'],
            'presets': ['ultra_fast', 'fast', 'standard', 'high_quality']
        },
        'your_tts': {
            'supports_emotions': False,
            'supports_voice_cloning': True,
            'supports_multilingual': True,
            'supports_speakers': False,
            'languages': ['en', 'fr', 'pt']
        },
        'standard': {
            'supports_emotions': False,
            'supports_voice_cloning': False,
            'supports_multilingual': False,
            'supports_speakers': False,
            'emotion_controls': [],
            'languages': ['en']
        }
    }

def _load_coqui_speakers():
    """Load pre-trained Coqui speakers for XTTS v2"""
    global coqui_speakers
    
    logger.info("Loading Coqui pre-trained speakers...")
    
    # Enhanced list of pre-defined Coqui speakers for XTTS v2
    coqui_speakers = {
        'Ana Florence': {
            'language': 'en', 
            'gender': 'female', 
            'description': 'Professional English female voice - clear and articulate',
            'accent': 'American',
            'age_range': 'Adult'
        },
        'Andrew Chipper': {
            'language': 'en', 
            'gender': 'male', 
            'description': 'Young English male voice - energetic and friendly',
            'accent': 'American',
            'age_range': 'Young Adult'
        },
        'Dionisio Schuyler': {
            'language': 'en', 
            'gender': 'male', 
            'description': 'Mature English male voice - authoritative and warm',
            'accent': 'American',
            'age_range': 'Middle-aged'
        },
        'Marcela Granados': {
            'language': 'es', 
            'gender': 'female', 
            'description': 'Spanish female voice - native speaker with clear pronunciation',
            'accent': 'Latin American',
            'age_range': 'Adult'
        },
        'Viktor Eka': {
            'language': 'de', 
            'gender': 'male', 
            'description': 'German male voice - professional and clear',
            'accent': 'Standard German',
            'age_range': 'Adult'
        },
        'Abrahan Mack': {
            'language': 'en', 
            'gender': 'male', 
            'description': 'Deep English male voice - rich and resonant',
            'accent': 'American',
            'age_range': 'Adult'
        },
        'Adde Michal': {
            'language': 'en', 
            'gender': 'male', 
            'description': 'Clear English male voice - crisp and professional',
            'accent': 'American',
            'age_range': 'Adult'
        },
        'Alexandra Hithe': {
            'language': 'en', 
            'gender': 'female', 
            'description': 'Elegant English female voice - sophisticated and smooth',
            'accent': 'British',
            'age_range': 'Adult'
        },
        'Alice Wonderland': {
            'language': 'en', 
            'gender': 'female', 
            'description': 'Whimsical English female voice - playful and expressive',
            'accent': 'American',
            'age_range': 'Young Adult'
        },
        'Claribel Dervla': {
            'language': 'en', 
            'gender': 'female', 
            'description': 'Articulate English female voice - precise and clear',
            'accent': 'Irish',
            'age_range': 'Adult'
        },
        'Elisabeth Ramsey': {
            'language': 'en', 
            'gender': 'female', 
            'description': 'Sophisticated English female voice - refined and elegant',
            'accent': 'British',
            'age_range': 'Middle-aged'
        },
        'Ernesto Bonnet': {
            'language': 'fr', 
            'gender': 'male', 
            'description': 'French male voice - native speaker with authentic accent',
            'accent': 'Parisian French',
            'age_range': 'Adult'
        }
    }
    
    logger.info(f"Loaded {len(coqui_speakers)} Coqui speakers")
    
    # Log available speakers
    for name, info in coqui_speakers.items():
        logger.info(f"  - {name}: {info['language'].upper()} {info['gender']} ({info['accent']})")

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

def get_or_load_model(model_name=None):
    """Get or load a specific TTS model with caching"""
    global current_tts_model, loaded_models
    
    if not TTS_AVAILABLE:
        # Return a mock model for demo purposes
        return _get_mock_tts_model()
    
    # Use default model if none specified
    if model_name is None:
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Check if model is already loaded
    if model_name in loaded_models:
        current_tts_model = loaded_models[model_name]
        return current_tts_model
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TTS(model_name).to(device)
        
        # Cache the loaded model
        loaded_models[model_name] = model
        current_tts_model = model
        
        logger.info(f"Loaded {model_name} on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        
        # Try fallback models
        fallback_models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts"
        ]
        
        for fallback in fallback_models:
            if fallback != model_name and fallback not in loaded_models:
                try:
                    model = TTS(fallback).to(device)
                    loaded_models[fallback] = model
                    current_tts_model = model
                    logger.info(f"Loaded fallback model: {fallback}")
                    return model
                except Exception as e2:
                    continue
        
        logger.error("Failed to load any TTS model")
        return None

def _get_mock_tts_model():
    """Create a mock TTS model for demo/testing when real TTS is unavailable"""
    class MockTTSModel:
        def __init__(self):
            self.model_name = "mock_tts_model"
            self.device = "cpu"
        
        def tts_to_file(self, text, file_path, **kwargs):
            """Generate a simple beep sound as placeholder"""
            try:
                import numpy as np
                
                # Log what voice/speaker was requested for demo purposes
                speaker = kwargs.get('speaker')
                speaker_wav = kwargs.get('speaker_wav')
                language = kwargs.get('language', 'en')
                
                if speaker:
                    logger.info(f"Mock TTS: Generating speech for speaker '{speaker}' in {language}")
                elif speaker_wav:
                    logger.info(f"Mock TTS: Generating speech with voice clone from {speaker_wav}")
                else:
                    logger.info(f"Mock TTS: Generating speech with default voice in {language}")
                
                # Generate a simple tone (beep) as placeholder
                sample_rate = 22050
                duration = min(len(text) * 0.1, 3.0)  # Duration based on text length, max 3 seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Vary frequency based on speaker/language for demo
                base_frequency = 440  # A4 note
                if speaker and 'female' in coqui_speakers.get(speaker, {}).get('gender', ''):
                    base_frequency = 523  # Higher pitch for female voices
                elif language == 'es':
                    base_frequency = 493  # Different tone for Spanish
                elif language == 'fr':
                    base_frequency = 466  # Different tone for French
                
                audio = 0.3 * np.sin(2 * np.pi * base_frequency * t)
                
                # Add some variation based on text length
                if len(text) > 50:
                    audio += 0.1 * np.sin(2 * np.pi * 880 * t)  # Add harmonic
                
                # Fade in/out to avoid clicks
                fade_samples = int(0.1 * sample_rate)
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                # Write audio file
                audio_int16 = (audio * 32767).astype(np.int16)
                
                if AUDIO_LIBS_AVAILABLE:
                    import soundfile as sf
                    sf.write(file_path, audio_int16, sample_rate)
                else:
                    # Fallback: write a simple WAV file
                    _write_simple_wav(file_path, audio_int16, sample_rate)
                
                logger.info(f"Generated mock audio for text: '{text[:50]}...' -> {file_path}")
                
            except Exception as e:
                logger.error(f"Error generating mock audio: {e}")
                # Create empty file as last resort
                with open(file_path, 'wb') as f:
                    f.write(b'')
    
    return MockTTSModel()

def _write_simple_wav(filename, audio_data, sample_rate):
    """Write a simple WAV file without external dependencies"""
    import struct
    
    with open(filename, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_data) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM format
        f.write(struct.pack('<H', 1))  # mono
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_data) * 2))
        
        # Audio data
        for sample in audio_data:
            f.write(struct.pack('<h', sample))

def get_current_model():
    """Get the current active model"""
    global current_tts_model
    if current_tts_model is None:
        return get_or_load_model()
    return current_tts_model

def get_model_type(model_name):
    """Determine the model type from model name"""
    if 'xtts' in model_name.lower():
        return 'xtts_v2'
    elif 'bark' in model_name.lower():
        return 'bark'
    elif 'tortoise' in model_name.lower():
        return 'tortoise'
    elif 'your_tts' in model_name.lower():
        return 'your_tts'
    else:
        return 'standard'

def _parse_unified_voice_id(voice_id):
    """Parse unified voice ID to get model and voice configuration"""
    try:
        # Get unified voices to find the matching voice
        response = get_unified_voices()
        if hasattr(response, 'get_json'):
            voices_data = response.get_json()
        else:
            voices_data = response
            
        if not voices_data.get('success'):
            return {}
            
        # Find the voice by ID
        for voice in voices_data.get('voices', []):
            if voice['id'] == voice_id:
                config = {
                    'model': voice['model'],
                    'language': voice['language']
                }
                
                if voice['type'] == 'coqui_speaker':
                    config['coqui_speaker'] = voice['speaker']
                elif voice['type'] == 'custom_voice':
                    config['voice_profile'] = voice['voice_profile']
                    
                return config
                
        return {}
    except Exception as e:
        logger.error(f"Error parsing unified voice ID {voice_id}: {e}")
        return {}

# Routes
@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Enhanced health check endpoint with detailed system info"""
    total_tts_models = _count_nested_models(available_models.get('tts_models', {}))
    total_coqui_speakers = len(coqui_speakers)
    
    # CPU info
    import os, platform, psutil
    cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # Device info
    device_info = "CPU"
    cuda_available = False
    try:
        if torch.cuda.is_available():
            cuda_available = True
            device_info = f"CUDA - {torch.cuda.get_device_name(0)}"
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.system(),
            'cpu_count': cpu_count,
            'cpu_usage_percent': cpu_usage,
            'memory_total_gb': round(memory.total / 1024**3, 2),
            'memory_available_gb': round(memory.available / 1024**3, 2),
            'memory_used_percent': memory.percent
        },
        'device': {
            'cuda_available': cuda_available,
            'device_name': device_info,
            'device_count': torch.cuda.device_count() if cuda_available else 0
        },
        'services': {
            'models_loaded': current_tts_model is not None,
            'tts_available': TTS_AVAILABLE,
            'audio_libs_available': AUDIO_LIBS_AVAILABLE,
            'model_manager_available': model_manager is not None
        },
        'models': {
            'total_tts_models': total_tts_models,
            'total_coqui_speakers': total_coqui_speakers,
            'loaded_models_count': len(loaded_models),
            'fallback_mode': not TTS_AVAILABLE
        }
    })

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    try:
        return jsonify({
            'tts_available': TTS_AVAILABLE,
            'audio_libs_available': AUDIO_LIBS_AVAILABLE,
            'available_models_structure': {
                'tts_models': {
                    'count': _count_nested_models(available_models.get('tts_models', {})),
                    'languages': list(available_models.get('tts_models', {}).keys())[:10],
                    'sample': _get_sample_models(available_models.get('tts_models', {}), 5)
                },
                'vocoder_models': {
                    'count': _count_nested_models(available_models.get('vocoder_models', {})),
                    'languages': list(available_models.get('vocoder_models', {}).keys())[:5]
                }
            },
            'coqui_speakers': {
                'count': len(coqui_speakers),
                'names': list(coqui_speakers.keys())[:10]
            },
            'voice_profiles': {
                'count': len(voice_profiles),
                'names': list(voice_profiles.keys())[:10]
            },
            'loaded_models': list(loaded_models.keys()),
            'current_model': getattr(current_tts_model, 'model_name', None) if current_tts_model else None,
            'model_capabilities_keys': list(model_capabilities.keys())
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def _get_sample_models(models_dict, limit=5):
    """Get a sample of available models for debugging"""
    samples = []
    count = 0
    
    for lang, datasets in models_dict.items():
        if count >= limit:
            break
        for dataset, models in datasets.items():
            if count >= limit:
                break
            for model_name in models.keys():
                if count >= limit:
                    break
                samples.append(f"tts_models/{lang}/{dataset}/{model_name}")
                count += 1
    
    return samples

@app.route('/api/models')
def get_models():
    """Get available TTS models with capabilities"""
    try:
        logger.info("API call to /api/models")
        logger.info(f"Available models keys: {list(available_models.keys())}")
        logger.info(f"TTS models count: {_count_nested_models(available_models.get('tts_models', {}))}")
        
        models_with_info = {}
        
        for category, models in available_models.items():
            if not isinstance(models, dict):
                continue
                
            models_with_info[category] = {}
            for lang, lang_models in models.items():
                if not isinstance(lang_models, dict):
                    continue
                    
                models_with_info[category][lang] = {}
                for dataset, dataset_models in lang_models.items():
                    if not isinstance(dataset_models, dict):
                        continue
                        
                    models_with_info[category][lang][dataset] = {}
                    for model_name, model_info in dataset_models.items():
                        full_model_name = f"{category}/{lang}/{dataset}/{model_name}"
                        model_type = get_model_type(full_model_name)
                        capabilities = model_capabilities.get(model_type, model_capabilities.get('standard', {}))
                        
                        # Ensure model_info is a dictionary
                        if not isinstance(model_info, dict):
                            model_info = {'description': str(model_info)}
                        
                        models_with_info[category][lang][dataset][model_name] = {
                            **model_info,
                            'capabilities': capabilities,
                            'full_name': full_model_name
                        }
        
        current_model_name = None
        if current_tts_model:
            current_model_name = getattr(current_tts_model, 'model_name', None)
        
        response_data = {
            'available_models': models_with_info,
            'current_model': current_model_name,
            'loaded_models': list(loaded_models.keys()),
            'model_capabilities': model_capabilities,
            'total_tts_models': _count_nested_models(available_models.get('tts_models', {})),
            'total_vocoder_models': _count_nested_models(available_models.get('vocoder_models', {})),
            'status': 'success'
        }
        
        logger.info(f"Returning {response_data['total_tts_models']} TTS models to frontend")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in /api/models endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'available_models': available_models,
            'current_model': None,
            'loaded_models': [],
            'model_capabilities': model_capabilities,
            'total_tts_models': 0,
            'total_vocoder_models': 0,
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/speakers')
def get_speakers():
    """Get available speakers (Coqui + custom voices)"""
    try:
        logger.info("API call to /api/speakers")
        logger.info(f"Coqui speakers count: {len(coqui_speakers)}")
        logger.info(f"Custom voices count: {len(voice_profiles)}")
        
        all_speakers = {
            'coqui_speakers': coqui_speakers,
            'custom_voices': voice_profiles,
            'total_count': len(coqui_speakers) + len(voice_profiles),
            'status': 'success'
        }
        
        # Log some speaker names for debugging
        if coqui_speakers:
            sample_speakers = list(coqui_speakers.keys())[:3]
            logger.info(f"Sample Coqui speakers: {sample_speakers}")
        
        return jsonify(all_speakers)
        
    except Exception as e:
        logger.error(f"Error in /api/speakers endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'coqui_speakers': {},
            'custom_voices': {},
            'total_count': 0,
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/models/<path:model_name>/info')
def get_model_info(model_name):
    """Get detailed information about a specific model"""
    try:
        # Load model to get detailed info
        model = get_or_load_model(model_name)
        if not model:
            return jsonify({'error': 'Model not found or failed to load'}), 404
        
        model_type = get_model_type(model_name)
        capabilities = model_capabilities.get(model_type, {})
        
        info = {
            'model_name': model_name,
            'model_type': model_type,
            'device': str(model.device) if hasattr(model, 'device') else 'Unknown',
            'capabilities': capabilities,
            'loaded': True
        }
        
        # Get model-specific information
        if hasattr(model, 'language_manager') and model.language_manager:
            info['supported_languages'] = list(model.language_manager.language_names)
        elif 'languages' in capabilities:
            info['supported_languages'] = capabilities['languages']
        
        if hasattr(model, 'speaker_manager') and model.speaker_manager:
            info['supported_speakers'] = list(model.speaker_manager.speaker_names)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voices')
def get_voices():
    """Get saved voice profiles"""
    return jsonify(voice_profiles)

@app.route('/api/unified_voices')
def get_unified_voices():
    """Get all available voices in one unified list for simplified UX"""
    try:
        unified_voices = []
        
        # Add TTS Model voices (featured models)
        featured_models = [
            {
                'id': 'xtts_v2_default',
                'name': '🎭 XTTS v2 - Voice Cloning Master',
                'type': 'tts_model',
                'model': 'tts_models/multilingual/multi-dataset/xtts_v2',
                'description': 'Best for voice cloning and emotional expression. Works in 16+ languages.',
                'capabilities': ['voice_cloning', 'emotions', 'multilingual'],
                'language': 'multilingual',
                'category': 'AI Models'
            },
            {
                'id': 'bark_default',
                'name': '😊 Bark - Emotional Storyteller',
                'type': 'tts_model', 
                'model': 'tts_models/multilingual/multi-dataset/bark',
                'description': 'Perfect for emotional, expressive speech with natural inflections.',
                'capabilities': ['emotions', 'multilingual'],
                'language': 'multilingual',
                'category': 'AI Models'
            },
            {
                'id': 'tortoise_default',
                'name': '🔄 Tortoise - High Quality English',
                'type': 'tts_model',
                'model': 'tts_models/en/multi-dataset/tortoise-v2', 
                'description': 'Ultra-high quality English voices with customizable presets.',
                'capabilities': ['voice_cloning', 'high_quality'],
                'language': 'en',
                'category': 'AI Models'
            }
        ]
        
        # Add quick language-specific models
        language_models = [
            {'id': 'en_fast', 'name': '⚡ English - Fast & Clear', 'model': 'tts_models/en/ljspeech/glow-tts', 'language': 'en'},
            {'id': 'es_spanish', 'name': '🇪🇸 Spanish Voice', 'model': 'tts_models/es/mai/tacotron2-DDC', 'language': 'es'},
            {'id': 'fr_french', 'name': '🇫🇷 French Voice', 'model': 'tts_models/fr/mai/tacotron2-DDC', 'language': 'fr'},
            {'id': 'de_german', 'name': '🇩🇪 German Voice', 'model': 'tts_models/de/thorsten/tacotron2-DDC', 'language': 'de'},
            {'id': 'it_italian', 'name': '🇮🇹 Italian Voice', 'model': 'tts_models/it/mai/tacotron2-DDC', 'language': 'it'},
            {'id': 'pt_portuguese', 'name': '🇵🇹 Portuguese Voice', 'model': 'tts_models/pt/cv/vits', 'language': 'pt'},
            {'id': 'ru_russian', 'name': '🇷🇺 Russian Voice', 'model': 'tts_models/ru/ruslan/tacotron2-DDC', 'language': 'ru'},
            {'id': 'zh_chinese', 'name': '🇨🇳 Chinese Voice', 'model': 'tts_models/zh/baker/tacotron2-DDC-GST', 'language': 'zh'},
            {'id': 'ja_japanese', 'name': '🇯🇵 Japanese Voice', 'model': 'tts_models/ja/kokoro/tacotron2-DDC', 'language': 'ja'},
            {'id': 'ko_korean', 'name': '🇰🇷 Korean Voice', 'model': 'tts_models/ko/kss/glow-tts', 'language': 'ko'}
        ]
        
        for model in language_models:
            featured_models.append({
                'id': model['id'],
                'name': model['name'],
                'type': 'tts_model',
                'model': model['model'],
                'description': f"Native {model['language'].upper()} voice optimized for clarity and speed.",
                'capabilities': ['fast', 'native_language'],
                'language': model['language'],
                'category': 'Language Models'
            })
        
        unified_voices.extend(featured_models)
        
        # Add Coqui pre-trained speakers
        logger.info(f"Adding {len(coqui_speakers)} Coqui speakers to unified voices")
        for speaker_name, speaker_info in coqui_speakers.items():
            speaker_voice = {
                'id': f'coqui_{speaker_name.lower().replace(" ", "_")}',
                'name': f'👤 {speaker_name}',
                'type': 'coqui_speaker',
                'speaker': speaker_name,
                'model': 'tts_models/multilingual/multi-dataset/xtts_v2',  # Coqui speakers use XTTS v2
                'description': speaker_info.get('description', f'{speaker_info.get("gender", "").title()} {speaker_info.get("language", "").upper()} speaker'),
                'capabilities': ['voice_cloning', 'emotions', 'multilingual'],
                'language': speaker_info.get('language', 'en'),
                'category': 'Pre-trained Speakers',
                'gender': speaker_info.get('gender'),
                'accent': speaker_info.get('accent')
            }
            unified_voices.append(speaker_voice)
            logger.debug(f"Added Coqui speaker: {speaker_name}")
        
        # Add custom voice profiles
        for voice_name, voice_info in voice_profiles.items():
            unified_voices.append({
                'id': f'custom_{voice_name.lower().replace(" ", "_")}',
                'name': f'🎤 {voice_name} (Custom)',
                'type': 'custom_voice',
                'voice_profile': voice_name,
                'model': 'tts_models/multilingual/multi-dataset/xtts_v2',  # Custom voices use XTTS v2
                'description': f'Your custom cloned voice: {voice_name}',
                'capabilities': ['voice_cloning', 'emotions', 'multilingual'],
                'language': 'multilingual',
                'category': 'Your Custom Voices',
                'created_at': voice_info.get('created_at')
            })
        
        # Group voices by category for easy selection
        grouped_voices = {}
        for voice in unified_voices:
            category = voice['category']
            if category not in grouped_voices:
                grouped_voices[category] = []
            grouped_voices[category].append(voice)
        
        return jsonify({
            'success': True,
            'voices': unified_voices,
            'grouped_voices': grouped_voices,
            'total_count': len(unified_voices),
            'categories': list(grouped_voices.keys())
        })
        
    except Exception as e:
        logger.error(f"Error getting unified voices: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'voices': [],
            'grouped_voices': {},
            'total_count': 0
        })

@app.route('/api/tts', methods=['POST'])
def synthesize_speech():
    """Enhanced TTS synthesis endpoint with emotion and model controls"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # New unified voice selection
        unified_voice_id = data.get('unified_voice_id')
        
        # Legacy support
        voice_profile = data.get('voice_profile')
        speaker_id = data.get('speaker_id')
        coqui_speaker = data.get('coqui_speaker')
        language = data.get('language', 'en')
        model_name = data.get('model_name')
        
        # Emotion and quality controls
        temperature = float(data.get('temperature', 0.7))
        repetition_penalty = float(data.get('repetition_penalty', 1.0))
        length_penalty = float(data.get('length_penalty', 1.0))
        preset = data.get('preset', 'standard')
        split_sentences = data.get('split_sentences', True)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Parse unified voice selection
        if unified_voice_id:
            voice_config = _parse_unified_voice_id(unified_voice_id)
            model_name = voice_config.get('model')
            voice_profile = voice_config.get('voice_profile') 
            coqui_speaker = voice_config.get('coqui_speaker')
            language = voice_config.get('language', language)
        
        # Load specific model if requested
        if model_name:
            model = get_or_load_model(model_name)
        else:
            model = get_or_load_model()
            
        if not model:
            return jsonify({'error': 'No TTS model available'}), 500
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(config.OUTPUTS_DIR, filename)
        
        # Determine model type for appropriate synthesis
        actual_model_name = getattr(model, 'model_name', model_name or 'unknown')
        
        # If using mock model but a specific model was requested, use that for type detection
        if actual_model_name == 'mock_tts_model' and model_name:
            model_type = get_model_type(model_name)
        else:
            model_type = get_model_type(actual_model_name)
            
        logger.info(f"Using model type: {model_type} for synthesis")
        
        # Synthesize based on model type and parameters
        synthesis_kwargs = {
            'text': text,
            'file_path': output_path
        }
        
        # Add language for multilingual models
        if model_capabilities.get(model_type, {}).get('supports_multilingual'):
            synthesis_kwargs['language'] = language
        
        # Voice selection based on model capabilities
        if voice_profile and voice_profile in voice_profiles:
            # Use custom cloned voice (works with XTTS v2)
            if model_type in ['xtts_v2', 'your_tts']:
                synthesis_kwargs['speaker_wav'] = voice_profiles[voice_profile]['audio_path']
            else:
                logger.warning(f"Custom voice not supported for model type: {model_type}")
        elif coqui_speaker and coqui_speaker in coqui_speakers:
            # Use Coqui pre-trained speaker (only works with XTTS v2)
            if model_type == 'xtts_v2':
                synthesis_kwargs['speaker'] = coqui_speaker
            else:
                logger.warning(f"Coqui speaker '{coqui_speaker}' not supported for model type: {model_type}. Using default voice.")
        elif speaker_id:
            # Use speaker ID for multi-speaker models
            model_caps = model_capabilities.get(model_type, {})
            if model_caps.get('supports_speakers'):
                synthesis_kwargs['speaker'] = speaker_id
            else:
                logger.warning(f"Speaker ID not supported for model type: {model_type}")
        
        # Add model-specific parameters
        if model_type == 'xtts_v2':
            if 'speaker_wav' in synthesis_kwargs or 'speaker' in synthesis_kwargs:
                synthesis_kwargs['split_sentences'] = split_sentences
                # Add emotion controls for XTTS v2
                if hasattr(model, 'tts_to_file'):
                    # These parameters might be available in advanced usage
                    pass
        elif model_type == 'bark':
            # Bark-specific parameters
            if temperature != 0.7:
                synthesis_kwargs['temperature'] = temperature
        elif model_type == 'tortoise':
            # Tortoise-specific parameters
            if preset != 'standard':
                synthesis_kwargs['preset'] = preset
        
        # Perform synthesis
        model.tts_to_file(**synthesis_kwargs)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'download_url': f'/api/download/{filename}',
            'model_used': getattr(model, 'model_name', 'unknown'),
            'model_type': model_type,
            'parameters_used': {
                'language': language,
                'temperature': temperature,
                'preset': preset,
                'split_sentences': split_sentences
            }
        })
        
    except Exception as e:
        logger.error(f"Enhanced TTS synthesis error: {e}")
        logger.error(traceback.format_exc())
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
    """Switch TTS model with enhanced capabilities"""
    global current_tts_model
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'No model name provided'}), 400
        
        # Load new model
        model = get_or_load_model(model_name)
        if not model:
            return jsonify({'error': f'Failed to load model: {model_name}'}), 500
        
        current_tts_model = model
        model_type = get_model_type(model_name)
        capabilities = model_capabilities.get(model_type, {})
        
        return jsonify({
            'success': True,
            'current_model': model_name,
            'model_type': model_type,
            'device': str(model.device) if hasattr(model, 'device') else 'unknown',
            'capabilities': capabilities
        })
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/presets')
def get_presets():
    """Get available presets for different model types"""
    presets = {
        'tortoise': {
            'ultra_fast': 'Fastest synthesis, lower quality',
            'fast': 'Fast synthesis, good quality',
            'standard': 'Balanced speed and quality',
            'high_quality': 'Best quality, slower synthesis'
        },
        'xtts_v2': {
            'default': 'Default XTTS v2 settings'
        },
        'bark': {
            'default': 'Default Bark settings'
        }
    }
    
    return jsonify({
        'presets': presets,
        'emotion_controls': {
            'temperature': {
                'min': 0.1,
                'max': 1.5,
                'default': 0.7,
                'description': 'Controls randomness and emotion in speech'
            },
            'repetition_penalty': {
                'min': 1.0,
                'max': 2.0,
                'default': 1.0,
                'description': 'Reduces repetitive speech patterns'
            },
            'length_penalty': {
                'min': 0.5,
                'max': 2.0,
                'default': 1.0,
                'description': 'Controls speech length and pacing'
            }
        }
    })

@app.route('/api/model/unload', methods=['POST'])
def unload_model():
    """Unload a specific model to free memory"""
    global loaded_models, current_tts_model
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'No model name provided'}), 400
        
        if model_name in loaded_models:
            del loaded_models[model_name]
            
            # If this was the current model, reset it
            if current_tts_model and getattr(current_tts_model, 'model_name', None) == model_name:
                current_tts_model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return jsonify({'success': True, 'message': f'Model {model_name} unloaded'})
        else:
            return jsonify({'error': 'Model not loaded'}), 404
            
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
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
        model = get_or_load_model()
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
        
        model = get_or_load_model()
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
def get_current_model_info():
    """Get detailed information about current model"""
    try:
        model = get_or_load_model()
        if not model:
            return jsonify({'error': 'No model loaded'}), 500
        
        model_name = getattr(model, 'model_name', 'Unknown')
        model_type = get_model_type(model_name)
        capabilities = model_capabilities.get(model_type, {})
        
        info = {
            'model_name': model_name,
            'model_type': model_type,
            'device': str(model.device) if hasattr(model, 'device') else 'Unknown',
            'capabilities': capabilities,
            'language_manager': getattr(model, 'language_manager', None) is not None,
            'speaker_manager': getattr(model, 'speaker_manager', None) is not None,
            'supported_languages': [],
            'supported_speakers': [],
            'memory_usage': _get_model_memory_usage()
        }
        
        # Get supported languages
        if hasattr(model, 'language_manager') and model.language_manager:
            info['supported_languages'] = list(model.language_manager.language_names)
        elif 'languages' in capabilities:
            info['supported_languages'] = capabilities['languages']
        
        # Get supported speakers
        if hasattr(model, 'speaker_manager') and model.speaker_manager:
            info['supported_speakers'] = list(model.speaker_manager.speaker_names)
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

def _get_model_memory_usage():
    """Get approximate memory usage information"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                'gpu_allocated_gb': round(allocated, 2),
                'gpu_cached_gb': round(cached, 2),
                'loaded_models_count': len(loaded_models)
            }
        else:
            return {
                'device': 'CPU',
                'loaded_models_count': len(loaded_models)
            }
    except Exception:
        return {'error': 'Unable to get memory info'}

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
        
        model = get_or_load_model()
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
        
        # Add metadata about the synthesis
        model_name = getattr(model, 'model_name', model_name or 'unknown')
        model_type = get_model_type(model_name)
        
        return jsonify({
            'success': True,
            'audio_url': f'/api/download/{filename}',
            'filename': filename,
            'text': text,
            'voice_id': voice_id,
            'language': language,
            'model_used': model_name,
            'model_type': model_type
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
    logger.info("🚀 Starting Enhanced Coqui TTS Web Server...")
    
    # Initialize models and voice profiles
    logger.info("🔧 Initializing models and voice profiles...")
    initialize_models()
    load_voice_profiles()
    
    # Log initialization results
    total_tts_models = _count_nested_models(available_models.get('tts_models', {}))
    total_coqui_speakers = len(coqui_speakers)
    
    logger.info(f"✅ Initialization complete:")
    logger.info(f"   - TTS models available: {total_tts_models}")
    logger.info(f"   - Coqui speakers: {total_coqui_speakers}")
    logger.info(f"   - Custom voice profiles: {len(voice_profiles)}")
    logger.info(f"   - CUDA available: {torch.cuda.is_available()}")
    logger.info(f"   - TTS library available: {TTS_AVAILABLE}")
    
    logger.info(f"🌐 Server starting on port {config.PORT}")
    logger.info(f"🔗 Access the web interface at: http://localhost:{config.PORT}")
    logger.info(f"🔍 Debug endpoint available at: http://localhost:{config.PORT}/debug")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=False,
        threaded=True
    )