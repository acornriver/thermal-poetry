#!/bin/bash
# Build script for Thermal Poetry

echo "Building Thermal Poetry..."

# 1. Install PyInstaller if not already installed
pip install pyinstaller

# 2. Build the app
pyinstaller --name "Thermal Poetry" \
    --windowed \
    --onefile \
    --add-data "log.md:." \
    --hidden-import pygame \
    --hidden-import sounddevice \
    --hidden-import numpy \
    --hidden-import escpos \
    --collect-all pygame \
    --icon NONE \
    main.py

echo "Build complete! Check the 'dist' folder for Thermal Poetry.app"
