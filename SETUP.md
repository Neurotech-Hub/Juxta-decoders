# Social Decoder Setup

## Virtual Environment Setup

The project uses a Python virtual environment to manage dependencies.

### Initial Setup
```bash
# Create virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Decoder

#### Option 1: Use the convenience script
```bash
./run_social_decoder.sh
```

#### Option 2: Manual activation
```bash
source .venv/bin/activate
python3 decoder_social.py
```

## Dependencies

- matplotlib>=3.7.0
- numpy>=1.24.0
- networkx>=3.0

## Output

The decoder generates:
- CSV files with event data
- Comprehensive analysis plots (2×3 grid layout)
- Individual timeline plots
- Original data files copied to output directory

All output files are saved to `./analysis_social/output/`
