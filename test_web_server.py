#!/usr/bin/env python3
"""
Quick test to verify the web server works on port 2201
"""

import os
import sys
import json
from flask import Flask, jsonify

# Create a minimal Flask app to test port 2201
app = Flask(__name__)

@app.route('/')
def index():
    return """
    <html>
        <head><title>Coqui TTS Test - Port 2201</title></head>
        <body>
            <h1>🐸 Coqui TTS Web Server Test</h1>
            <p>Successfully running on port 2201!</p>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/api/test">API Test</a></li>
            </ul>
        </body>
    </html>
    """

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'port': 2201,
        'message': 'Test server is running successfully'
    })

@app.route('/api/test')
def api_test():
    return jsonify({
        'success': True,
        'message': 'API endpoint is working on port 2201',
        'features': [
            'Text-to-Speech',
            'Voice Cloning',
            'File Management',
            'Multi-language Support'
        ]
    })

if __name__ == '__main__':
    print("🐸 Starting test server on port 2201...")
    app.run(host='0.0.0.0', port=2201, debug=True)