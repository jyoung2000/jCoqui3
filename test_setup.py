#!/usr/bin/env python3
"""
Test script to verify setup.py can read all requirements files
"""
import os
import sys

def test_requirements_files():
    """Test that all requirements files can be read"""
    print("🔍 Testing requirements file reading...")
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Test function from setup.py
    def read_requirements_file(filename):
        """Read requirements file if it exists, return empty list otherwise"""
        try:
            with open(os.path.join(cwd, filename), "r") as f:
                lines = f.readlines()
                print(f"  ✅ {filename}: {len(lines)} lines")
                return lines
        except FileNotFoundError:
            print(f"  ❌ {filename}: not found")
            return []
    
    # Test all requirements files
    files_to_test = [
        "requirements.txt",
        "requirements.notebooks.txt", 
        "requirements.dev.txt",
        "requirements.ja.txt",
        "requirements.web.txt"
    ]
    
    all_good = True
    for filename in files_to_test:
        lines = read_requirements_file(filename)
        if filename in ["requirements.txt"] and len(lines) == 0:
            print(f"  ⚠️  {filename} is empty - this might be a problem")
            all_good = False
    
    if all_good:
        print("✅ All requirements files can be read successfully!")
    else:
        print("❌ Some requirements files have issues")
        all_good = False
    
    # Test README.md reading
    print("\n🔍 Testing README.md reading...")
    try:
        with open("README.md", "r", encoding="utf-8") as readme_file:
            readme_content = readme_file.read()
            print(f"  ✅ README.md: {len(readme_content)} characters")
    except FileNotFoundError:
        print("  ❌ README.md: not found")
        all_good = False
    
    return all_good

def test_setup_imports():
    """Test that setup.py can import and run"""
    print("\n🔍 Testing setup.py imports...")
    
    try:
        # Test the core dependencies needed by setup.py
        import numpy
        print("  ✅ numpy imported successfully")
        
        from Cython.Build import cythonize
        print("  ✅ Cython imported successfully")
        
        from setuptools import Extension, find_packages, setup
        print("  ✅ setuptools imported successfully")
        
        print("✅ All setup.py dependencies available!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🐸 Coqui TTS Setup Verification")
    print("=" * 40)
    
    success1 = test_requirements_files()
    success2 = test_setup_imports()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 All tests passed! setup.py should work correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the issues above.")
        sys.exit(1)