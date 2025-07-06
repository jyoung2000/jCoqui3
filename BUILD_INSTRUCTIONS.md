# Build and Deployment Instructions

## Building the Docker Container

### For Testing (CPU-only)

1. Build the container:
```bash
docker-compose build
```

2. Run the container:
```bash
docker-compose up -d
```

3. Check logs:
```bash
docker-compose logs -f
```

4. Access the web interface:
- Open browser to: http://localhost:2201
- API documentation: http://localhost:2201/api/v1/status

### For Production (with GPU)

1. Create a GPU-enabled Dockerfile:
```bash
cp Dockerfile Dockerfile.gpu
```

2. Edit Dockerfile.gpu first line:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as builder
```

3. Add GPU support to docker-compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

4. Build and run:
```bash
docker-compose build
docker-compose up -d
```

## Testing the Container

### 1. Health Check
```bash
curl http://localhost:2201/health
```

Expected response:
```json
{
  "status": "healthy",
  "cuda_available": false,
  "models_loaded": true
}
```

### 2. Test Text-to-Speech
```bash
curl -X POST http://localhost:2201/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the Coqui TTS system.", "language": "en"}'
```

### 3. Test Voice Cloning
1. Upload a voice sample through the web interface
2. Or use the API:
```bash
curl -X POST http://localhost:2201/api/v1/clone \
  -F "audio=@voice_sample.wav" \
  -F "name=test_voice"
```

### 4. List Available Voices
```bash
curl http://localhost:2201/api/v1/voices
```

## Troubleshooting Build Issues

### Out of Memory
- Increase Docker memory limit to at least 4GB
- Use `--no-cache` flag: `docker-compose build --no-cache`

### Network Timeouts
- Use build args for pip:
```bash
docker-compose build --build-arg PIP_DEFAULT_TIMEOUT=100
```

### Permission Denied
- On Linux, add user to docker group:
```bash
sudo usermod -aG docker $USER
```

### Slow Build
- Use BuildKit:
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

## Deployment on UnRAID

1. Copy files to UnRAID server
2. Navigate to Docker tab
3. Add new container:
   - Repository: coqui-tts-web
   - WebUI: http://[IP]:2201/
   - Port: 2201

4. Add paths:
   - /app/models -> /mnt/user/appdata/coqui-tts/models
   - /app/voices -> /mnt/user/appdata/coqui-tts/voices
   - /app/outputs -> /mnt/user/appdata/coqui-tts/outputs

5. Memory limit: 4096MB minimum

## Performance Optimization

### CPU Mode
- Use smaller models (tacotron2 instead of XTTS)
- Reduce batch size
- Enable model caching

### GPU Mode
- Ensure CUDA drivers are installed
- Use `--gpus all` flag
- Monitor GPU memory with `nvidia-smi`

### Model Download
First run will download models (~2GB). To pre-download:
```bash
docker run -it coqui-tts-web python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

## Security Considerations

1. Run as non-root user (already configured)
2. Limit file upload size (10MB default)
3. Use reverse proxy for HTTPS
4. Implement rate limiting for API

## Monitoring

Check container stats:
```bash
docker stats coqui-tts-web
```

View logs:
```bash
docker-compose logs -f --tail=100
```

## Backup

Backup voice profiles and outputs:
```bash
docker run --rm -v tts-voices:/voices -v tts-outputs:/outputs \
  -v $(pwd):/backup alpine tar czf /backup/coqui-backup.tar.gz /voices /outputs
```