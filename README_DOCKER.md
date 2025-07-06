# J-Coqui Web Enhanced 🐸

An enhanced Docker-based web interface for Coqui TTS with advanced voice cloning, model management, and API integration capabilities.

## Features ✨

### 🎙️ Text-to-Speech
- Support for 16+ languages
- Multiple TTS models (XTTS v2, Tacotron2, VITS, etc.)
- Real-time synthesis with web interface
- Batch processing capabilities
- Granular voice controls (speed, pitch, volume)

### 🎭 Voice Cloning
- One-shot voice cloning with minimal samples
- Voice profile management and storage
- High-quality voice reproduction
- Support for multiple audio formats (WAV, MP3, FLAC)

### 🔄 Voice Conversion
- Convert existing audio to different voices
- Real-time voice transformation
- Preserve speech patterns and emotions

### 🌐 Web Interface
- Modern, responsive UI with Bootstrap
- Real-time audio playback
- Drag-and-drop file uploads
- Progress tracking and status monitoring
- File management and downloads

### 🚀 API Integration
- RESTful API for external applications
- Comprehensive endpoints for all features
- JSON and form-data support
- Status monitoring and health checks

### 🐳 Docker Support
- Optimized multi-stage builds
- Lightweight Python base images
- Persistent data storage with volumes
- Memory and resource management
- Health checks and graceful shutdown

## Quick Start 🚀

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM recommended
- NVIDIA GPU (optional, for CUDA acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/jyoung2000/j-coqui-web-enhanced.git
cd j-coqui-web-enhanced
```

2. **Build and start the container:**
```bash
docker compose build
docker compose up -d
```

3. **Access the web interface:**
Open your browser to `http://localhost:2201`

The container will automatically:
- Download required TTS models
- Set up the web interface
- Initialize voice cloning capabilities

## Configuration ⚙️

### Environment Variables
- `PORT`: Web server port (default: 2201)
- `TTS_MODELS_DIR`: Model storage directory
- `TTS_VOICES_DIR`: Voice profiles directory
- `TTS_OUTPUTS_DIR`: Generated audio directory
- `TTS_UPLOADS_DIR`: Upload temporary directory

### Volume Mounts
- `./data/models`: Persistent model storage
- `./data/voices`: Voice profile storage
- `./data/outputs`: Generated audio files
- `./data/uploads`: Temporary uploads

## API Documentation 📚

### Synthesis Endpoint
```bash
POST /api/v1/synthesize
Content-Type: application/json

{
  "text": "Hello, world!",
  "voice_id": "my_voice",
  "language": "en",
  "format": "wav"
}
```

### Voice Cloning Endpoint
```bash
POST /api/v1/clone
Content-Type: multipart/form-data

audio=@voice_sample.wav
name=my_custom_voice
```

### List Voices
```bash
GET /api/v1/voices
```

### Service Status
```bash
GET /api/v1/status
```

## Advanced Usage 🔧

### Custom Models
Place custom models in `/app/models/` directory inside the container or mount them via volumes.

### Batch Processing
Use the `/api/batch_synthesis` endpoint for processing multiple texts at once.

### Voice Conversion
Convert existing audio files to different voices using the voice conversion API.

## Performance Optimization 🏃

### GPU Acceleration
The container supports NVIDIA CUDA for faster processing:
```yaml
services:
  coqui-tts-web:
    # ... other config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Memory Management
Adjust memory limits in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
```

## Troubleshooting 🔧

### Common Issues

1. **Container fails to start:**
   - Check available memory (4GB+ recommended)
   - Verify port 2201 is not in use
   - Check Docker daemon is running

2. **Model download fails:**
   - Ensure internet connectivity
   - Check disk space for model storage
   - Verify write permissions on volume mounts

3. **Audio processing errors:**
   - Verify audio file formats are supported
   - Check file size limits (100MB default)
   - Ensure audio sample rate compatibility

### Logs
View container logs:
```bash
docker compose logs -f coqui-tts-web
```

### Health Check
Check service status:
```bash
curl http://localhost:2201/health
```

## Development 🛠️

### Local Development
For development purposes, you can run without Docker:

```bash
# Install dependencies
pip install -r requirements.web.txt

# Set environment variables
export PYTHONPATH="/path/to/project:$PYTHONPATH"
export NUMBA_CACHE_DIR="/tmp/numba_cache"

# Run the application
python web_server/app.py
```

### Building Custom Images
Modify `Dockerfile.web` for custom requirements and rebuild:
```bash
docker compose build --no-cache
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License 📄

This project is built upon Coqui TTS and maintains compatibility with its licensing terms. Please refer to the original Coqui TTS license for details.

## Acknowledgments 🙏

- [Coqui TTS](https://github.com/coqui-ai/TTS) - The underlying TTS engine
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Bootstrap](https://getbootstrap.com/) - UI framework

---

**Built with ❤️ for the TTS community**