#!/usr/bin/env python3
"""
Enhanced Coqui TTS Web Server Startup Script
Handles initialization, model management, and robust error handling
"""

import os
import sys
import time
import signal
import logging
import subprocess
from pathlib import Path

# Add TTS to path before importing
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        '/app/models',
        '/app/voices', 
        '/app/outputs',
        '/app/uploads',
        '/tmp/numba_cache',
        '/app/models/.cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def setup_environment():
    """Setup environment variables and configurations"""
    os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
    os.environ['NUMBA_DISABLE_JIT'] = '0'
    os.environ['TTS_CACHE_DIR'] = '/app/models/.cache'
    os.environ['PYTHONPATH'] = '/app:' + os.environ.get('PYTHONPATH', '')
    
    logger.info("Environment variables configured")

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        import flask
        logger.info(f"Flask version: {flask.__version__}")
        
        # Try importing TTS components
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        logger.info("TTS components imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

def download_default_models():
    """Download essential models for immediate functionality"""
    try:
        from TTS.api import TTS
        
        # Download XTTS v2 for voice cloning (if not already cached)
        logger.info("Checking for XTTS v2 model...")
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        
        try:
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            logger.info("XTTS v2 model ready")
        except Exception as e:
            logger.warning(f"Could not load XTTS v2: {e}")
            # Fallback to simpler model
            try:
                tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                logger.info("Fallback Tacotron2 model ready")
            except Exception as e2:
                logger.error(f"Could not load fallback model: {e2}")
                
    except Exception as e:
        logger.error(f"Model download failed: {e}")

def run_health_check():
    """Perform basic health check"""
    logger.info("Running system health check...")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('/app')
    logger.info(f"Disk space - Total: {total//1024//1024//1024}GB, "
                f"Used: {used//1024//1024//1024}GB, "
                f"Free: {free//1024//1024//1024}GB")
    
    # Check memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemAvailable:' in line:
                    mem_available = int(line.split()[1]) // 1024  # Convert to MB
                    logger.info(f"Available memory: {mem_available}MB")
                    break
    except:
        logger.warning("Could not read memory info")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function"""
    logger.info("🐸 Starting Enhanced Coqui TTS Web Server...")
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Setup environment
        setup_environment()
        
        # Ensure directories exist
        ensure_directories()
        
        # Run health check
        run_health_check()
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Dependency check failed. Exiting.")
            sys.exit(1)
        
        # Download default models
        download_default_models()
        
        # Import and start the Flask app
        logger.info("Starting Flask application...")
        
        # Change to app directory to ensure relative imports work
        os.chdir('/app')
        
        # Import the Flask app
        from web_server.app import app, config
        
        logger.info(f"🚀 Server starting on port {config.PORT}")
        logger.info("✅ Coqui TTS Web Interface is ready!")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=config.PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        logger.exception("Detailed error information:")
        sys.exit(1)

if __name__ == '__main__':
    main()