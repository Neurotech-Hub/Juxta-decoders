# JUXTA Data Decoders

This project provides Python scripts to decode and analyze data from JUXTA device files, including ADC burst data and social interaction data.

## Features

### ADC Data Decoder (`decode_adc.py`)
- Decodes ADC burst records from binary format
- Extracts timing information and sample data
- Converts raw samples to voltage values
- Supports multiple event types (timer burst, single event, peri-event)
- Generates multiple plot types:
  - Overlaid event data
  - Individual event plots
  - Summary statistics
- Saves plots as JPG files in the `./analysis_adc/figures` directory

### Social Data Decoder (`decoder_social.py`)
- Decodes social interaction records from binary format
- Extracts device scan data, motion detection, and environmental sensors
- Resolves MAC addresses using MACIDX files
- Generates multiple plot types:
  - Device detection timeline
  - Motion detection timeline
  - Battery and temperature monitoring
  - **RSSI timeline with device proximity analysis**
  - Summary statistics
- **Exports data to CSV files** for further analysis
- Saves plots as JPG files in the `./analysis_social/figures` directory

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

### Running the Scripts

#### ADC Data Decoder
1. **From VS Code**: 
   - Open the project in VS Code
   - Open `decode_adc.py`
   - Click the "Play" button (▶️) in the top-right corner of the editor

2. **From Terminal**:
   ```bash
   # Make sure virtual environment is activated
   source .venv/bin/activate
   
   # Run the ADC decoder
   python3 decode_adc.py
   ```

#### Social Data Decoder
1. **From VS Code**: 
   - Open the project in VS Code
   - Open `decoder_social.py`
   - Click the "Play" button (▶️) in the top-right corner of the editor

2. **From Terminal**:
   ```bash
   # Make sure virtual environment is activated
   source .venv/bin/activate
   
   # Run the social data decoder
   python3 decoder_social.py
   ```

#### Analysis Mode
Both decoders support comprehensive analysis mode:

```bash
python3 decode_adc.py --header
python3 decoder_social.py --header
```

#### Time Format Options (Social Decoder)
The social decoder supports different time formats:

```bash
# Default: Local timestamps (e.g., "2025-09-09 17:43:00")
python3 decoder_social.py

# Simple time format (e.g., "17:43")
python3 decoder_social.py --simple-time
python3 decoder_social.py -s
```

### Line Plot Visualizations

The social data decoder generates comprehensive **line plots** that show data trends over time:

- **Device Timeline**: Device detections over time
- **Motion Timeline**: Motion events per minute over time  
- **Battery & Temperature**: System health monitoring over time
- **RSSI Timeline**: Signal strength for each device over time
- **Summary Plots**: Record types, device counts, motion counts, and average RSSI over time
- **Reference lines** (RSSI Timeline): 
  - Green dashed line at -30 dBm (very close)
  - Orange dashed line at -60 dBm (close)
  - Red dashed line at -90 dBm (far)

These line plots are perfect for analyzing:
- **Social interaction patterns** - when devices were nearby
- **Proximity trends** - how close devices were over time
- **Device movement** - signal strength changes indicating movement
- **System health** - battery and temperature monitoring
- **Activity patterns** - motion and device detection trends
- **Temporal analysis** - social activity patterns throughout the day

### Working with CSV Data

The CSV files use **JSON encoding** for device data, making them easy to work with in Python:

```python
import csv
import json

# Read the CSV
with open('social_events.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Parse JSON-encoded device data
        device_macs = json.loads(row['device_macs']) if row['device_macs'] else []
        device_rssis = json.loads(row['device_rssis']) if row['device_rssis'] else []
        
        # Now you have clean arrays: ['FF0123', 'BABE99'] and [-73, -62]
        print(f"Time: {row['time']}, Devices: {device_macs}, RSSIs: {device_rssis}")
```

See `example_csv_usage.py` for a complete example of analyzing the CSV data.

### Output

The scripts will:
1. Decode the most recent data file in their respective directories
2. Print a summary of the decoded data to the console
3. Generate plots and save them to their respective directories:

**ADC Data Plots** (`./analysis_adc/figures/`):
   - `all_events_overlay.jpg` - All events overlaid on one plot
   - `timer_burst_01.jpg`, `single_event_01.jpg`, etc. - Individual event plots
   - `event_summary.jpg` - Summary statistics

**Social Data Plots** (`./analysis_social/figures/`):
   - `device_timeline.jpg` - Device detection timeline
   - `motion_timeline.jpg` - Motion detection timeline
   - `battery_temperature.jpg` - Battery and temperature monitoring
   - `rssi_timeline.jpg` - **RSSI timeline with device proximity analysis**
   - `social_summary.jpg` - Summary statistics

**CSV Data Export** (`./analysis_social/`):
   - `system_events.csv` - System events (boot, connection, settings, errors)
   - `social_events.csv` - Social interaction data with **JSON-encoded device arrays**
     - `device_macs`: JSON array of device MAC addresses (e.g., `["FF0123", "BABE99"]`)
     - `device_rssis`: JSON array of corresponding RSSI values (e.g., `[-73, -62]`)

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
