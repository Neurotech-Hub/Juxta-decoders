#!/usr/bin/env python3
"""
Simple test script for the ADC decoder
Run this from VS Code using the play button
"""

import os
import sys

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adc_decoder import main

if __name__ == "__main__":
    print("Running ADC Decoder Test...")
    main()
