#!/usr/bin/env python3
"""
Simple health check script for the container
"""
import sys
import requests

def check_health():
    try:
        response = requests.get('http://localhost:2201/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("Health check passed")
                return 0
            else:
                print(f"Health check failed: {data}")
                return 1
        else:
            print(f"Health check failed: HTTP {response.status_code}")
            return 1
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(check_health())