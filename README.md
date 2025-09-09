# ADC File Decoder

This project provides a Python script to decode ADC burst data from JUXTA device files and generate visualization plots.

## Features

- Decodes ADC burst records from binary format
- Extracts timing information and sample data
- Converts raw samples to voltage values
- Generates multiple plot types:
  - Overlaid burst data
  - Individual burst plots
  - Summary statistics
- Saves plots as JPG files in the `./figures` directory

## Setup Instructions for macOS

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 1. Create Virtual Environment

Open Terminal and navigate to the project directory:

```bash
cd /Users/mattgaidica/Documents/Software/Juxta/decoders
```

Create a virtual environment in the `.venv` directory:

```bash
python3 -m venv .venv
```

### 2. Activate Virtual Environment

Activate the virtual environment:

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

This will install:
- matplotlib (for plotting)
- numpy (for numerical operations)

### 4. Verify Installation

Test that everything is working:

```bash
python adc_decoder.py
```

## Usage

### Running the Script

1. **From VS Code**: 
   - Open the project in VS Code
   - Open `adc_decoder.py`
   - Click the "Play" button (▶️) in the top-right corner of the editor

2. **From Terminal**:
   ```bash
   # Make sure virtual environment is activated
   source .venv/bin/activate
   
   # Run the script
   python adc_decoder.py
   ```

### Output

The script will:
1. Decode the test file `hublink_file_content_20250908081219_250908.txt`
2. Print a summary of the decoded data to the console
3. Generate plots and save them to the `./figures/` directory:
   - `all_bursts_overlay.jpg` - All bursts overlaid on one plot
   - `burst_01.jpg`, `burst_02.jpg`, etc. - Individual burst plots
   - `burst_summary.jpg` - Summary statistics

## File Format

The decoder expects ADC files with the following format:
- **Header**: 12 bytes (timestamp, microsecond offset, sample count, duration)
- **Sample Data**: N bytes (where N = sample count from header)
- **Format**: Big-endian binary data

See `spec_ADC_debug.md` for detailed format specification.

## Troubleshooting

### Common Issues

1. **"No module named 'matplotlib'"**
   - Make sure the virtual environment is activated
   - Run `pip install -r requirements.txt` again

2. **"Test file not found"**
   - Ensure `hublink_file_content_20250908081219_250908.txt` is in the project directory
   - Check the filename matches exactly

3. **Permission errors**
   - Make sure you have write permissions to the `./figures` directory
   - The script will create the directory if it doesn't exist

### Deactivating Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
decoders/
├── .venv/                    # Virtual environment (created during setup)
├── figures/                  # Output directory for plots
├── adc_decoder.py           # Main decoder script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── spec_ADC_debug.md       # ADC format specification
└── hublink_file_content_20250908081219_250908.txt  # Test data file
```

## Dependencies

- **matplotlib**: For creating plots and visualizations
- **numpy**: For numerical operations and array handling
- **struct**: Built-in Python module for binary data parsing
- **datetime**: Built-in Python module for timestamp handling
