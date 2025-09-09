#!/usr/bin/env python3
"""
ADC File Format Decoder

This script decodes ADC burst data from JUXTA device files and generates plots.
Based on the specification in spec_ADC_debug.md
"""

import struct
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict, Any


def find_valid_headers(file_data: bytes) -> List[Dict[str, Any]]:
    """
    Find all valid headers in the file data using robust detection
    
    Args:
        file_data: Raw file data (bytes)
        
    Returns:
        list: List of valid header information
    """
    valid_headers = []
    offset = 0
    
    while offset < len(file_data) - 12:
        # Look for the known header pattern (68Bx...)
        if file_data[offset] == 0x68 and (file_data[offset+1] & 0xF0) == 0xB0:
            try:
                header = file_data[offset:offset + 12]
                unix_timestamp, microsecond_offset, sample_count, duration_us = struct.unpack('>IIHH', header)
                
                # Validate header values
                if (microsecond_offset <= 999999 and sample_count > 0 and sample_count < 10000 and 
                    duration_us > 0 and duration_us < 100000 and 
                    unix_timestamp > 1000000000 and unix_timestamp < 2000000000):
                    
                    valid_headers.append({
                        'offset': offset,
                        'unix_timestamp': unix_timestamp,
                        'microsecond_offset': microsecond_offset,
                        'sample_count': sample_count,
                        'duration_us': duration_us,
                        'raw_header': header.hex()
                    })
                    
            except (struct.error, IndexError):
                pass
        
        offset += 1
    
    return valid_headers


def decode_adc_burst(file_data: bytes, header_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode a single ADC burst record from file data using header info
    
    Args:
        file_data: Raw file data (bytes)
        header_info: Header information from find_valid_headers
        
    Returns:
        dict: Decoded burst information
    """
    offset = header_info['offset']
    sample_count = header_info['sample_count']
    
    # Extract sample data
    sample_start = offset + 12
    sample_end = sample_start + sample_count
    
    # Handle truncated data
    if sample_end > len(file_data):
        print(f"Warning: Not enough data for {sample_count} samples, only {len(file_data) - sample_start} bytes available")
        sample_count = len(file_data) - sample_start
        sample_end = len(file_data)
    
    raw_samples = file_data[sample_start:sample_end]
    
    # Convert to voltage
    voltage_samples = [(s / 255.0) * 4000.0 - 2000.0 for s in raw_samples]
    
    # Calculate absolute timestamp
    absolute_timestamp = header_info['unix_timestamp'] + (header_info['microsecond_offset'] / 1_000_000.0)
    
    return {
        'unix_timestamp': header_info['unix_timestamp'],
        'microsecond_offset': header_info['microsecond_offset'],
        'sample_count': sample_count,
        'duration_us': header_info['duration_us'],
        'absolute_timestamp': absolute_timestamp,
        'raw_samples': raw_samples,
        'voltage_samples': voltage_samples,
        'next_offset': sample_end
    }


def decode_adc_file(filename: str) -> List[Dict[str, Any]]:
    """
    Decode entire ADC file using robust header detection
    
    Args:
        filename: Path to ADC file
        
    Returns:
        list: List of decoded burst records
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    # Find all valid headers first
    valid_headers = find_valid_headers(file_data)
    
    # Decode each burst using header information
    bursts = []
    for header_info in valid_headers:
        try:
            burst = decode_adc_burst(file_data, header_info)
            bursts.append(burst)
        except Exception as e:
            print(f"Warning: Failed to decode burst at offset {header_info['offset']}: {e}")
    
    return bursts


def plot_burst_data(bursts: List[Dict[str, Any]], output_dir: str = "./figures") -> None:
    """
    Generate plots for ADC burst data - focusing on data section with header info in titles
    
    Args:
        bursts: List of decoded burst records
        output_dir: Directory to save plot files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files in the figures directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {filename}")
    
    # Plot 1: All bursts overlaid (data only)
    plt.figure(figsize=(12, 8))
    for i, burst in enumerate(bursts):
        # Create time axis based on actual data length and duration
        time_axis = np.linspace(0, burst['duration_us'] / 1000.0, len(burst['voltage_samples']))
        plt.plot(time_axis, burst['voltage_samples'], alpha=0.7, label=f'Burst {i+1}')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('ADC Data - All Bursts Overlaid\n(Data section only, header info in individual plots)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_bursts_overlay.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Individual burst plots (data only with header info in title)
    for i, burst in enumerate(bursts):
        plt.figure(figsize=(10, 6))
        
        # Create time axis based on actual data length
        time_axis = np.linspace(0, burst['duration_us'] / 1000.0, len(burst['voltage_samples']))
        plt.plot(time_axis, burst['voltage_samples'], 'b-', linewidth=1)
        
        # Add comprehensive header info to title
        timestamp_str = datetime.fromtimestamp(burst['absolute_timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Calculate effective sampling rate
        duration_seconds = burst['duration_us'] / 1_000_000.0
        sampling_rate = len(burst['voltage_samples']) / duration_seconds if duration_seconds > 0 else 0
        
        plt.title(f'ADC Data - Burst {i+1}\n'
                 f'Timestamp: {timestamp_str}\n'
                 f'Header: Unix={burst["unix_timestamp"]}, Micro={burst["microsecond_offset"]}μs\n'
                 f'Data: {len(burst["voltage_samples"])} samples, Duration: {burst["duration_us"]}μs\n'
                 f'Sampling Rate: {sampling_rate:.0f} Hz ({sampling_rate/1000:.1f} kHz)\n'
                 f'Voltage Range: {min(burst["voltage_samples"]):.1f} to {max(burst["voltage_samples"]):.1f} mV')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'burst_{i+1:02d}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Data analysis summary
    if len(bursts) > 1:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Sampling rates
        plt.subplot(2, 2, 1)
        sampling_rates = []
        for burst in bursts:
            duration_seconds = burst['duration_us'] / 1_000_000.0
            sampling_rate = len(burst['voltage_samples']) / duration_seconds if duration_seconds > 0 else 0
            sampling_rates.append(sampling_rate)
        
        plt.bar(range(1, len(sampling_rates) + 1), sampling_rates)
        plt.xlabel('Burst Number')
        plt.ylabel('Sampling Rate (Hz)')
        plt.title('Effective Sampling Rate per Burst')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Durations from headers
        plt.subplot(2, 2, 2)
        durations = [burst['duration_us'] for burst in bursts]
        plt.bar(range(1, len(durations) + 1), durations)
        plt.xlabel('Burst Number')
        plt.ylabel('Duration (μs)')
        plt.title('Duration per Burst (from header)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Voltage ranges (data analysis)
        plt.subplot(2, 2, 3)
        voltage_mins = [min(burst['voltage_samples']) for burst in bursts]
        voltage_maxs = [max(burst['voltage_samples']) for burst in bursts]
        x_pos = range(1, len(bursts) + 1)
        plt.bar(x_pos, voltage_maxs, alpha=0.7, label='Max Voltage')
        plt.bar(x_pos, voltage_mins, alpha=0.7, label='Min Voltage')
        plt.xlabel('Burst Number')
        plt.ylabel('Voltage (mV)')
        plt.title('Voltage Range per Burst (data analysis)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Burst timing (from headers)
        plt.subplot(2, 2, 4)
        timestamps = [burst['absolute_timestamp'] for burst in bursts]
        time_diffs = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
        plt.plot(range(1, len(time_diffs) + 1), time_diffs, 'o-')
        plt.xlabel('Burst Number')
        plt.ylabel('Time from First Burst (s)')
        plt.title('Burst Timing (from headers)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'burst_summary.jpg'), dpi=300, bbox_inches='tight')
        plt.close()


def print_burst_summary(bursts: List[Dict[str, Any]]) -> None:
    """
    Print a summary of decoded burst data
    
    Args:
        bursts: List of decoded burst records
    """
    print(f"\n=== ADC Data Processing Summary ===")
    print(f"Total bursts found: {len(bursts)}")
    print(f"Total data samples: {sum(len(burst['voltage_samples']) for burst in bursts)}")
    
    if bursts:
        print(f"\nFirst burst timestamp: {datetime.fromtimestamp(bursts[0]['absolute_timestamp'])}")
        print(f"Last burst timestamp: {datetime.fromtimestamp(bursts[-1]['absolute_timestamp'])}")
        
        print(f"\nBurst Details (Header + Data Analysis):")
        for i, burst in enumerate(bursts):
            timestamp_str = datetime.fromtimestamp(burst['absolute_timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            actual_samples = len(burst['voltage_samples'])
            
            # Calculate effective sampling rate
            duration_seconds = burst['duration_us'] / 1_000_000.0
            sampling_rate = actual_samples / duration_seconds if duration_seconds > 0 else 0
            
            print(f"  Burst {i+1}: {timestamp_str}")
            print(f"    Header: Unix={burst['unix_timestamp']}, Micro={burst['microsecond_offset']}μs")
            print(f"    Data: {actual_samples} samples, Duration: {burst['duration_us']}μs")
            print(f"    Sampling Rate: {sampling_rate:.0f} Hz ({sampling_rate/1000:.1f} kHz)")
            print(f"    Voltage: {min(burst['voltage_samples']):.1f} to {max(burst['voltage_samples']):.1f} mV")
            print()


def main():
    """
    Main function to decode ADC file and generate plots
    """
    # Find the only .txt file (excluding requirements.txt)
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
    
    print("ADC File Decoder")
    print("================")
    
    if not txt_files:
        print("Error: No ADC data files found!")
        print("Please ensure there's a .txt file with ADC data in the current directory.")
        return
    
    if len(txt_files) > 1:
        print(f"Multiple .txt files found: {txt_files}")
        print("Using the first one...")
    
    test_file = txt_files[0]
    
    try:
        # Decode the file
        print(f"Decoding file: {test_file}")
        bursts = decode_adc_file(test_file)
        
        if not bursts:
            print("No valid bursts found in the file.")
            return
        
        # Print summary
        print_burst_summary(bursts)
        
        # Generate plots
        print(f"\nGenerating plots...")
        plot_burst_data(bursts)
        print(f"Plots saved to ./figures/ directory")
        
        print("\nDecoding complete!")
        
    except Exception as e:
        print(f"Error during decoding: {e}")
        raise


if __name__ == "__main__":
    main()
