#!/bin/bash
# Script to run the social decoder with virtual environment activated

# Activate virtual environment
source .venv/bin/activate

# Run the decoder with all arguments passed through
python3 decoder_social.py "$@"
