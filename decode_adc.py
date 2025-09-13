#!/usr/bin/env python3
"""
ADC File Format Decoder and Header Analyzer

This script decodes ADC burst data from JUXTA device files and generates plots.
It also provides comprehensive header analysis and validation capabilities.
Based on the specification in spec_ADC_debug.md

Combined functionality from adc_header.py for comprehensive analysis.
"""

import struct
import os
import re
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional imports for plotting functionality
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/numpy not available. Plotting functionality disabled.")


def find_adc_file(analysis_dir: str) -> Optional[str]:
    """
    Find the single ADC file in the directory
    
    Args:
        analysis_dir: Directory to search for .txt files
        
    Returns:
        str: Path to the ADC file, or None if no valid file found
    """
    if not os.path.exists(analysis_dir):
        return None
    
    txt_files = [f for f in os.listdir(analysis_dir) if f.endswith('.txt')]
    
    if not txt_files:
        return None
    
    # Pattern to match JX filename format: JX_DEVICEID_YYMMDD.txt
    jx_pattern = re.compile(r'JX_([A-Z0-9]+)_(\d{6})\.txt')
    
    valid_files = []
    for filename in txt_files:
        match = jx_pattern.match(filename)
        if match:
            device_id = match.group(1)
            recording_date_str = match.group(2)
            try:
                # Parse recording date: YYMMDD
                year = 2000 + int(recording_date_str[:2])
                month = int(recording_date_str[2:4])
                day = int(recording_date_str[4:6])
                recording_date = datetime(year, month, day)
                valid_files.append((recording_date, filename, device_id))
            except ValueError:
                # Invalid date format, skip this file
                continue
    
    if len(valid_files) == 0:
        print("Error: No valid JX_DEVICEID_YYMMDD.txt files found!")
        return None
    elif len(valid_files) > 1:
        print("Error: Multiple ADC files found in directory:")
        for _, filename, device_id in valid_files:
            print(f"  - {filename} (Device: {device_id})")
        print("Only one ADC file is allowed per analysis directory.")
        return None
    
    # Return the single valid file
    return os.path.join(analysis_dir, valid_files[0][1])


def extract_device_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract device ID from JX filename format
    
    Args:
        filename: Filename in format JX_DEVICEID_YYMMDD.txt
        
    Returns:
        str: Device ID, or None if parsing fails
    """
    pattern = re.compile(r'JX_([A-Z0-9]+)_\d{6}\.txt')
    match = pattern.match(os.path.basename(filename))
    
    if match:
        return match.group(1)
    
    return None


def find_valid_headers(file_data: bytes) -> List[Dict[str, Any]]:
    """
    Find all valid headers in the file data using 13-byte header format
    
    Args:
        file_data: Raw file data (bytes)
        
    Returns:
        list: List of valid header information
    """
    valid_headers = []
    offset = 0
    
    while offset < len(file_data) - 13:
        try:
            # Parse 13-byte header at current offset
            header = file_data[offset:offset + 13]
            unix_timestamp, microsecond_offset, sample_count, duration_us, event_type = struct.unpack('>IIHHB', header)
            
            # Basic validation
            if (microsecond_offset <= 999999 and 
                duration_us > 0 and duration_us < 100000 and 
                unix_timestamp > 0 and
                event_type in [0x00, 0x01, 0x02]):  # Valid event types
                
                valid_headers.append({
                    'offset': offset,
                    'unix_timestamp': unix_timestamp,
                    'microsecond_offset': microsecond_offset,
                    'sample_count': sample_count,
                    'duration_us': duration_us,
                    'event_type': event_type,
                    'raw_header': header.hex()
                })
                
                # Move to next potential header position
                # For single events (event_type = 0x02), next header is at offset + 16
                # For other events, next header is at offset + 13 + sample_count
                if event_type == 0x02:  # Single event
                    offset += 16
                else:
                    offset += 13 + sample_count
            else:
                # Invalid header, try next byte
                offset += 1
                
        except (struct.error, IndexError):
            # Can't parse header at this position, try next byte
            offset += 1
    
    return valid_headers


def analyze_adc_headers(filename: str) -> List[Dict[str, Any]]:
    """
    Analyze ADC file headers and data sections - comprehensive ground truth checker
    
    Args:
        filename: Path to ADC file (hex text format)
        
    Returns:
        list: List of header analysis results
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    print(f"File Analysis: {filename}")
    print(f"Hex data length: {len(hex_data)} characters")
    print(f"Binary data length: {len(file_data)} bytes")
    print("=" * 60)
    
    headers = []
    discrepancies = []
    offset = 0
    potential_headers = []
    
    # First pass: Find all potential headers (every 13-byte alignment)
    print("SCANNING FOR POTENTIAL HEADERS...")
    while offset < len(file_data) - 13:
        try:
            header = file_data[offset:offset + 13]
            unix_timestamp, microsecond_offset, sample_count, duration_us, event_type = struct.unpack('>IIHHB', header)
            
            potential_headers.append({
                'offset': offset,
                'unix_timestamp': unix_timestamp,
                'microsecond_offset': microsecond_offset,
                'sample_count': sample_count,
                'duration_us': duration_us,
                'event_type': event_type,
                'raw_header': header.hex()
            })
            
            # Move to next potential header position based on event type
            if event_type == 0x02:  # Single event
                offset += 16
            else:
                offset += 13 + sample_count
            
        except (struct.error, IndexError):
            offset += 1
    
    print(f"Found {len(potential_headers)} potential headers at 12-byte boundaries")
    print()
    
    # Second pass: Validate and categorize headers
    print("VALIDATING HEADERS...")
    valid_headers = []
    invalid_headers = []
    
    for i, ph in enumerate(potential_headers):
        # Check validation criteria
        issues = []
        
        if ph['microsecond_offset'] > 999999:
            issues.append(f"Invalid microsecond offset: {ph['microsecond_offset']} (>999999)")
        
        if ph['sample_count'] <= 0 or ph['sample_count'] > 10000:
            issues.append(f"Invalid sample count: {ph['sample_count']} (should be 1-10000)")
        
        if ph['duration_us'] <= 0 or ph['duration_us'] > 100000:
            issues.append(f"Invalid duration: {ph['duration_us']} μs (should be 1-100000)")
        
        if ph['unix_timestamp'] <= 0:
            issues.append(f"Invalid timestamp: {ph['unix_timestamp']} (should be positive)")
        
        # Validate event type
        if ph['event_type'] not in [0x00, 0x01, 0x02]:
            issues.append(f"Invalid event type: 0x{ph['event_type']:02X} (should be 0x00, 0x01, or 0x02)")
        
        # Check if this looks like sample data (all same value) for waveform events
        if ph['event_type'] in [0x00, 0x01] and ph['sample_count'] > 0 and ph['sample_count'] < 10000:
            data_start = ph['offset'] + 13
            data_end = min(data_start + ph['sample_count'], len(file_data))
            if data_end > data_start:
                sample_data = file_data[data_start:data_end]
                unique_values = len(set(sample_data))
                if unique_values == 1:
                    issues.append(f"All sample data is same value: 0x{sample_data[0]:02X}")
                elif unique_values < 10 and len(sample_data) > 100:
                    issues.append(f"Very limited sample variation: only {unique_values} unique values")
        
        if issues:
            invalid_headers.append({
                'offset': ph['offset'],
                'issues': issues,
                'raw_header': ph['raw_header']
            })
        else:
            valid_headers.append(ph)
    
    # Third pass: Analyze valid headers for data completeness and timing
    print("ANALYZING VALID HEADERS...")
    for i, vh in enumerate(valid_headers):
        # Calculate data section based on event type
        if vh['event_type'] == 0x02:  # Single event
            data_start = vh['offset'] + 13
            data_end = data_start + 3  # 3 bytes for single event data
            expected_data_length = 3
            actual_data_length = min(3, len(file_data) - data_start)
        else:  # Timer burst or peri-event
            data_start = vh['offset'] + 13
            data_end = data_start + vh['sample_count']
            expected_data_length = vh['sample_count']
            actual_data_length = min(vh['sample_count'], len(file_data) - data_start)
        
        # Calculate absolute timestamp
        absolute_timestamp = vh['unix_timestamp'] + (vh['microsecond_offset'] / 1_000_000.0)
        
        # Determine event type name
        event_type_names = {0x00: 'timer_burst', 0x01: 'peri_event', 0x02: 'single_event'}
        event_type_name = event_type_names.get(vh['event_type'], 'unknown')
        
        header_info = {
            'event_number': i + 1,
            'event_type': event_type_name,
            'event_type_code': vh['event_type'],
            'offset': vh['offset'],
            'unix_timestamp': vh['unix_timestamp'],
            'microsecond_offset': vh['microsecond_offset'],
            'sample_count': vh['sample_count'],
            'duration_us': vh['duration_us'],
            'absolute_timestamp': absolute_timestamp,
            'data_start': data_start,
            'data_end': data_end,
            'expected_data_length': expected_data_length,
            'actual_data_length': actual_data_length,
            'data_complete': actual_data_length == expected_data_length,
            'next_header_offset': data_end
        }
        
        headers.append(header_info)
        
        # Check for discrepancies
        if not header_info['data_complete']:
            discrepancies.append(f"Event {i+1}: Incomplete data - {actual_data_length}/{expected_data_length} bytes")
        
        # Check timing consistency
        if i > 0:
            prev_timestamp = headers[i-1]['absolute_timestamp']
            time_diff = absolute_timestamp - prev_timestamp
            if time_diff < 0:
                discrepancies.append(f"Event {i+1}: Negative time difference from previous event: {time_diff:.3f}s")
            elif time_diff > 3600:  # More than 1 hour
                discrepancies.append(f"Event {i+1}: Large time gap from previous event: {time_diff:.1f}s")
        
        # Print header analysis
        print(f"Event {header_info['event_number']} ({event_type_name}):")
        print(f"  📍 FILE DATA (Raw from file):")
        print(f"    Header offset: {vh['offset']} bytes")
        print(f"    Raw header hex: {file_data[vh['offset']:vh['offset']+13].hex()}")
        print(f"    Unix timestamp: {vh['unix_timestamp']} (raw 32-bit value)")
        print(f"    Microsecond offset: {vh['microsecond_offset']} (raw 32-bit value)")
        print(f"    Sample count: {vh['sample_count']} (raw 16-bit value)")
        print(f"    Duration: {vh['duration_us']} μs (raw 16-bit value)")
        print(f"    Event type: 0x{vh['event_type']:02X} ({event_type_name})")
        print(f"  🧮 CALCULATED VALUES:")
        print(f"    Human timestamp: {datetime.fromtimestamp(vh['unix_timestamp'])} (calculated from Unix timestamp)")
        print(f"    Absolute timestamp: {absolute_timestamp:.6f} (Unix + microsecond_offset/1M)")
        print(f"    Data section: bytes {data_start}-{data_end-1} (calculated: offset+13 to offset+13+data_length)")
        print(f"    Expected data length: {expected_data_length} bytes (from event type and sample_count)")
        print(f"    Actual data available: {actual_data_length} bytes (calculated: min(expected, file_size-data_start))")
        print(f"    Data complete: {'YES' if header_info['data_complete'] else 'NO'} (calculated: actual==expected)")
        print(f"    Next header at: {data_end} bytes (calculated: data_start + data_length)")
        print()
    
    # Report invalid headers
    if invalid_headers:
        print("INVALID HEADERS FOUND:")
        for ih in invalid_headers:
            print(f"  Offset {ih['offset']}: {ih['raw_header']}")
            for issue in ih['issues']:
                print(f"    - {issue}")
        print()
    
    # Store discrepancies for summary
    if headers:
        headers[0]['discrepancies'] = discrepancies
    
    return headers


def print_header_summary(headers: List[Dict[str, Any]]) -> None:
    """
    Print summary of header analysis with discrepancy reporting
    
    Args:
        headers: List of header analysis results
    """
    if not headers:
        print("No valid headers found.")
        return
    
    print("=" * 60)
    print("GROUND TRUTH SUMMARY")
    print("=" * 60)
    print(f"📊 FILE ANALYSIS RESULTS:")
    print(f"   Total valid bursts found: {len(headers)} (calculated from header validation)")
    print(f"   Total expected samples: {sum(h['sample_count'] for h in headers)} (sum of sample_count fields)")
    print(f"   Total actual data bytes: {sum(h['actual_data_length'] for h in headers)} (calculated from file size)")
    
    # Data completeness analysis
    incomplete_bursts = [h for h in headers if not h['data_complete']]
    if incomplete_bursts:
        print(f"\n❌ DATA COMPLETENESS ISSUES:")
        print(f"   Bursts with incomplete data: {len(incomplete_bursts)}")
        for burst in incomplete_bursts:
            print(f"   - Burst {burst['burst_number']}: {burst['actual_data_length']}/{burst['expected_data_length']} bytes")
    else:
        print(f"\n✅ DATA COMPLETENESS: All {len(headers)} bursts have complete data")
    
    # Timestamp validation
    invalid_timestamps = [h for h in headers if h['microsecond_offset'] > 999999]
    if invalid_timestamps:
        print(f"\n❌ TIMESTAMP ISSUES:")
        print(f"   Bursts with invalid microsecond offsets: {len(invalid_timestamps)}")
        for burst in invalid_timestamps:
            print(f"   - Burst {burst['burst_number']}: {burst['microsecond_offset']} μs")
    else:
        print(f"\n✅ TIMESTAMP VALIDATION: All {len(headers)} bursts have valid timestamps")
    
    # Time range analysis
    if len(headers) > 1:
        first_time = headers[0]['absolute_timestamp']
        last_time = headers[-1]['absolute_timestamp']
        time_span = last_time - first_time
        print(f"\n📊 TIMING ANALYSIS (Calculated from timestamps):")
        print(f"   Time span: {time_span:.3f} seconds (calculated: last_timestamp - first_timestamp)")
        print(f"   First burst: {datetime.fromtimestamp(first_time)} (calculated from Unix timestamp)")
        print(f"   Last burst: {datetime.fromtimestamp(last_time)} (calculated from Unix timestamp)")
        
        # Check burst intervals
        intervals = []
        for i in range(1, len(headers)):
            interval = headers[i]['absolute_timestamp'] - headers[i-1]['absolute_timestamp']
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            print(f"   Average burst interval: {avg_interval:.1f} seconds (calculated from timestamp differences)")
            
            # Check for consistent intervals (should be ~5 seconds based on spec)
            expected_interval = 5.0
            interval_variance = max(intervals) - min(intervals)
            if interval_variance > 1.0:  # More than 1 second variance
                print(f"   ⚠️  WARNING: High interval variance ({interval_variance:.1f}s) - expected ~{expected_interval}s (from spec)")
            else:
                print(f"   ✅ Consistent burst intervals (~{avg_interval:.1f}s) (calculated)")
    
    # Report discrepancies
    discrepancies = headers[0].get('discrepancies', [])
    if discrepancies:
        print(f"\n🚨 DISCREPANCIES DETECTED:")
        for discrepancy in discrepancies:
            print(f"   - {discrepancy}")
    else:
        print(f"\n✅ NO DISCREPANCIES: Data appears consistent with specification")
    
    # Specification compliance check
    print(f"\n📋 SPECIFICATION COMPLIANCE:")
    spec_compliant = True
    
    # Check sample count (should be 1000 per spec)
    non_1000_samples = [h for h in headers if h['sample_count'] != 1000]
    if non_1000_samples:
        print(f"   ❌ Sample count: {len(non_1000_samples)} bursts don't have 1000 samples")
        spec_compliant = False
    else:
        print(f"   ✅ Sample count: All {len(headers)} bursts have 1000 samples")
    
    # Check duration range (should be ~5ms per spec)
    duration_issues = []
    for h in headers:
        duration_ms = h['duration_us'] / 1000.0
        if duration_ms < 4.0 or duration_ms > 6.0:
            duration_issues.append(f"Burst {h['burst_number']}: {duration_ms:.1f}ms")
    
    if duration_issues:
        print(f"   ❌ Duration: {len(duration_issues)} bursts outside 4-6ms range")
        for issue in duration_issues:
            print(f"      - {issue}")
        spec_compliant = False
    else:
        print(f"   ✅ Duration: All {len(headers)} bursts within 4-6ms range")
    
    # Overall compliance
    if spec_compliant:
        print(f"\n🎯 OVERALL: Data fully compliant with specification")
    else:
        print(f"\n⚠️  OVERALL: Data has specification compliance issues")


def decode_adc_event(file_data: bytes, header_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode a single ADC event record from file data using header info
    Supports multiple event types: timer burst, single event, peri-event
    
    Args:
        file_data: Raw file data (bytes)
        header_info: Header information from find_valid_headers
        
    Returns:
        dict: Decoded event information
    """
    offset = header_info['offset']
    sample_count = header_info['sample_count']
    event_type_code = header_info['event_type']
    
    # Calculate absolute timestamp
    absolute_timestamp = header_info['unix_timestamp'] + (header_info['microsecond_offset'] / 1_000_000.0)
    
    # Determine event type based on event_type_code
    if event_type_code == 0x02:  # Single event
        # Single event - read 3-byte event data
        event_start = offset + 13
        event_end = event_start + 3
        
        if event_end > len(file_data):
            print(f"Warning: Not enough data for single event, only {len(file_data) - event_start} bytes available")
            return None
            
        event_data = file_data[event_start:event_end]
        peak_positive, peak_negative, reserved = struct.unpack('BBB', event_data)
        
        return {
            'event_type': 'single_event',
            'event_type_code': event_type_code,
            'unix_timestamp': header_info['unix_timestamp'],
            'microsecond_offset': header_info['microsecond_offset'],
            'duration_us': header_info['duration_us'],
            'absolute_timestamp': absolute_timestamp,
            'peak_positive': peak_positive,
            'peak_negative': peak_negative,
            'peak_positive_mv': (peak_positive / 255.0) * 4000.0 - 2000.0,
            'peak_negative_mv': (peak_negative / 255.0) * 4000.0 - 2000.0,
            'next_offset': event_end
        }
    else:
        # Timer burst or peri-event - read sample data
        sample_start = offset + 13
        sample_end = sample_start + sample_count
        
        # Handle truncated data
        if sample_end > len(file_data):
            print(f"Warning: Not enough data for {sample_count} samples, only {len(file_data) - sample_start} bytes available")
            sample_count = len(file_data) - sample_start
            sample_end = len(file_data)
        
        raw_samples = file_data[sample_start:sample_end]
        
        # Convert to voltage
        voltage_samples = [(s / 255.0) * 4000.0 - 2000.0 for s in raw_samples]
        
        # Determine event type name based on event_type_code
        event_type_names = {0x00: 'timer_burst', 0x01: 'peri_event'}
        event_type_name = event_type_names.get(event_type_code, 'unknown')
        
        return {
            'event_type': event_type_name,
            'event_type_code': event_type_code,
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
    Supports multiple event types: timer burst, single event, peri-event
    
    Args:
        filename: Path to ADC file
        
    Returns:
        list: List of decoded event records
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    # Find all valid headers first
    valid_headers = find_valid_headers(file_data)
    
    # Decode each event using header information
    events = []
    for header_info in valid_headers:
        try:
            event = decode_adc_event(file_data, header_info)
            if event is not None:
                events.append(event)
        except Exception as e:
            print(f"Warning: Failed to decode event at offset {header_info['offset']}: {e}")
    
    return events


def plot_event_data(events: List[Dict[str, Any]], device_id: str = "", output_dir: str = "./analysis_adc/figures") -> None:
    """
    Generate plots for ADC event data - supports multiple event types
    
    Args:
        events: List of decoded event records
        device_id: Device ID for plot titles
        output_dir: Directory to save plot files
    """
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting functionality requires matplotlib and numpy.")
        print("Please install with: pip install matplotlib numpy")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files in the figures directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {filename}")
    
    # Separate events by type
    waveform_events = [e for e in events if e['event_type'] in ['timer_burst', 'peri_event']]
    single_events = [e for e in events if e['event_type'] == 'single_event']
    
    # Plot 1: All waveform events overlaid (if any)
    if waveform_events:
        plt.figure(figsize=(12, 8))
        for i, event in enumerate(waveform_events):
            # Create time axis based on actual data length and duration
            time_axis = np.linspace(0, event['duration_us'] / 1000.0, len(event['voltage_samples']))
            plt.plot(time_axis, event['voltage_samples'], alpha=0.7, 
                    label=f'{event["event_type"].replace("_", " ").title()} {i+1}')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        device_title = f" (Device: {device_id})" if device_id else ""
        plt.title(f'ADC Data - All Waveform Events Overlaid{device_title}\n(Data section only, header info in individual plots)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_events_overlay.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Individual waveform event plots
    for i, event in enumerate(waveform_events):
        plt.figure(figsize=(10, 6))
        
        # Create time axis based on actual data length
        time_axis = np.linspace(0, event['duration_us'] / 1000.0, len(event['voltage_samples']))
        plt.plot(time_axis, event['voltage_samples'], 'b-', linewidth=1)
        
        # Add comprehensive header info to title
        timestamp_str = datetime.fromtimestamp(event['absolute_timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Calculate effective sampling rate
        duration_seconds = event['duration_us'] / 1_000_000.0
        sampling_rate = len(event['voltage_samples']) / duration_seconds if duration_seconds > 0 else 0
        
        event_type_display = event['event_type'].replace('_', ' ').title()
        device_title = f" (Device: {device_id})" if device_id else ""
        plt.title(f'ADC Data - {event_type_display} {i+1}{device_title}\n'
                 f'Timestamp: {timestamp_str}\n'
                 f'Header: Unix={event["unix_timestamp"]}, Micro={event["microsecond_offset"]}μs\n'
                 f'Data: {len(event["voltage_samples"])} samples, Duration: {event["duration_us"]}μs\n'
                 f'Sampling Rate: {sampling_rate:.0f} Hz ({sampling_rate/1000:.1f} kHz)\n'
                 f'Voltage Range: {min(event["voltage_samples"]):.1f} to {max(event["voltage_samples"]):.1f} mV')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{event["event_type"]}_{i+1:02d}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Data analysis summary
    if len(events) > 1:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Event type distribution
        plt.subplot(2, 2, 1)
        event_types = [e['event_type'] for e in events]
        type_counts = {}
        for event_type in event_types:
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        plt.bar(range(len(type_counts)), list(type_counts.values()))
        plt.xticks(range(len(type_counts)), [t.replace('_', ' ').title() for t in type_counts.keys()], rotation=45)
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.title('Event Type Distribution')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Durations from headers
        plt.subplot(2, 2, 2)
        durations = [e['duration_us'] for e in events]
        plt.bar(range(1, len(durations) + 1), durations)
        plt.xlabel('Event Number')
        plt.ylabel('Duration (μs)')
        plt.title('Duration per Event (from header)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Voltage ranges (for waveform events only)
        if waveform_events:
            plt.subplot(2, 2, 3)
            voltage_mins = [min(e['voltage_samples']) for e in waveform_events]
            voltage_maxs = [max(e['voltage_samples']) for e in waveform_events]
            x_pos = range(1, len(waveform_events) + 1)
            plt.bar(x_pos, voltage_maxs, alpha=0.7, label='Max Voltage')
            plt.bar(x_pos, voltage_mins, alpha=0.7, label='Min Voltage')
            plt.xlabel('Waveform Event Number')
            plt.ylabel('Voltage (mV)')
            plt.title('Voltage Range per Waveform Event')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Show single event peak values
            plt.subplot(2, 2, 3)
            if single_events:
                peak_pos = [e['peak_positive_mv'] for e in single_events]
                peak_neg = [e['peak_negative_mv'] for e in single_events]
                x_pos = range(1, len(single_events) + 1)
                plt.bar(x_pos, peak_pos, alpha=0.7, label='Peak Positive')
                plt.bar(x_pos, peak_neg, alpha=0.7, label='Peak Negative')
                plt.xlabel('Single Event Number')
                plt.ylabel('Voltage (mV)')
                plt.title('Peak Values per Single Event')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # Subplot 4: Event timing (from headers)
        plt.subplot(2, 2, 4)
        timestamps = [e['absolute_timestamp'] for e in events]
        time_diffs = [timestamps[i] - timestamps[0] for i in range(len(timestamps))]
        plt.plot(range(1, len(time_diffs) + 1), time_diffs, 'o-')
        plt.xlabel('Event Number')
        plt.ylabel('Time from First Event (s)')
        plt.title('Event Timing (from headers)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'event_summary.jpg'), dpi=300, bbox_inches='tight')
        plt.close()


def print_event_summary(events: List[Dict[str, Any]]) -> None:
    """
    Print a summary of decoded event data
    
    Args:
        events: List of decoded event records
    """
    print(f"\n=== ADC Data Processing Summary ===")
    print(f"Total events found: {len(events)}")
    
    # Count by event type
    event_type_counts = {}
    for event in events:
        event_type = event['event_type']
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
    
    print(f"Event type breakdown:")
    for event_type, count in event_type_counts.items():
        print(f"  {event_type.replace('_', ' ').title()}: {count}")
    
    # Count total samples (waveform events only)
    waveform_events = [e for e in events if e['event_type'] in ['timer_burst', 'peri_event']]
    total_samples = sum(len(e['voltage_samples']) for e in waveform_events)
    print(f"Total waveform samples: {total_samples}")
    
    if events:
        print(f"\nFirst event timestamp: {datetime.fromtimestamp(events[0]['absolute_timestamp'])}")
        print(f"Last event timestamp: {datetime.fromtimestamp(events[-1]['absolute_timestamp'])}")
        
        print(f"\nEvent Details (Header + Data Analysis):")
        for i, event in enumerate(events):
            timestamp_str = datetime.fromtimestamp(event['absolute_timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            event_type_display = event['event_type'].replace('_', ' ').title()
            
            print(f"  Event {i+1} ({event_type_display}): {timestamp_str}")
            print(f"    Header: Unix={event['unix_timestamp']}, Micro={event['microsecond_offset']}μs")
            print(f"    Duration: {event['duration_us']}μs")
            
            if event['event_type'] == 'single_event':
                print(f"    Peak Positive: {event['peak_positive_mv']:.1f} mV (raw: {event['peak_positive']})")
                print(f"    Peak Negative: {event['peak_negative_mv']:.1f} mV (raw: {event['peak_negative']})")
            else:
                actual_samples = len(event['voltage_samples'])
                # Calculate effective sampling rate
                duration_seconds = event['duration_us'] / 1_000_000.0
                sampling_rate = actual_samples / duration_seconds if duration_seconds > 0 else 0
                
                print(f"    Data: {actual_samples} samples")
                print(f"    Sampling Rate: {sampling_rate:.0f} Hz ({sampling_rate/1000:.1f} kHz)")
                print(f"    Voltage: {min(event['voltage_samples']):.1f} to {max(event['voltage_samples']):.1f} mV")
            print()


def export_adc_data_to_csv(events: List[Dict[str, Any]], output_dir: str = "./analysis_adc") -> None:
    """
    Export ADC event data to CSV file
    
    Args:
        events: List of decoded ADC events
        output_dir: Directory to save CSV files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if not events:
        print("No events to export.")
        return
    
    # Export all events to a single CSV file
    adc_csv_path = os.path.join(output_dir, 'adc_events.csv')
    
    with open(adc_csv_path, 'w', newline='') as csvfile:
        # Define fieldnames for all event types
        fieldnames = [
            'event_number', 'event_type', 'event_type_code', 'timestamp', 'datetime',
            'unix_timestamp', 'microsecond_offset', 'duration_us', 'sample_count',
            'peak_positive', 'peak_negative', 'peak_positive_mv', 'peak_negative_mv',
            'raw_samples', 'voltage_samples', 'voltage_min_mv', 'voltage_max_mv',
            'voltage_mean_mv', 'voltage_std_mv', 'sampling_rate_hz'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, event in enumerate(events):
            # Calculate derived statistics for waveform events
            voltage_min_mv = None
            voltage_max_mv = None
            voltage_mean_mv = None
            voltage_std_mv = None
            sampling_rate_hz = None
            
            if event['event_type'] in ['timer_burst', 'peri_event']:
                voltage_samples = event['voltage_samples']
                if voltage_samples:
                    voltage_min_mv = min(voltage_samples)
                    voltage_max_mv = max(voltage_samples)
                    voltage_mean_mv = sum(voltage_samples) / len(voltage_samples)
                    
                    # Calculate standard deviation
                    if len(voltage_samples) > 1:
                        variance = sum((v - voltage_mean_mv) ** 2 for v in voltage_samples) / (len(voltage_samples) - 1)
                        voltage_std_mv = variance ** 0.5
                    
                    # Calculate sampling rate
                    duration_seconds = event['duration_us'] / 1_000_000.0
                    if duration_seconds > 0:
                        sampling_rate_hz = len(voltage_samples) / duration_seconds
            
            # Format datetime string
            datetime_str = datetime.fromtimestamp(event['absolute_timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Prepare row data
            row_data = {
                'event_number': i + 1,
                'event_type': event['event_type'],
                'event_type_code': f"0x{event['event_type_code']:02X}",
                'timestamp': event['absolute_timestamp'],
                'datetime': datetime_str,
                'unix_timestamp': event['unix_timestamp'],
                'microsecond_offset': event['microsecond_offset'],
                'duration_us': event['duration_us'],
                'sample_count': event.get('sample_count', 0),
                'peak_positive': event.get('peak_positive', ''),
                'peak_negative': event.get('peak_negative', ''),
                'peak_positive_mv': event.get('peak_positive_mv', ''),
                'peak_negative_mv': event.get('peak_negative_mv', ''),
                'raw_samples': json.dumps(list(event.get('raw_samples', []))) if 'raw_samples' in event else '',
                'voltage_samples': json.dumps(event.get('voltage_samples', [])) if 'voltage_samples' in event else '',
                'voltage_min_mv': voltage_min_mv if voltage_min_mv is not None else '',
                'voltage_max_mv': voltage_max_mv if voltage_max_mv is not None else '',
                'voltage_mean_mv': voltage_mean_mv if voltage_mean_mv is not None else '',
                'voltage_std_mv': voltage_std_mv if voltage_std_mv is not None else '',
                'sampling_rate_hz': sampling_rate_hz if sampling_rate_hz is not None else ''
            }
            
            writer.writerow(row_data)
    
    print(f"ADC events exported to: {adc_csv_path}")
    print(f"Exported {len(events)} events with complete header and sample data")
    print(f"Note: raw_samples and voltage_samples are JSON-encoded arrays for easy parsing")


def main():
    """
    Main function to decode ADC file and generate plots, with optional header analysis
    """
    import sys
    
    # Look for the most recent juxta file in the analysis directory
    analysis_dir = './analysis_adc'
    if not os.path.exists(analysis_dir):
        print(f"Analysis directory '{analysis_dir}' not found!")
        return
    
    print("ADC File Decoder and Header Analyzer")
    print("====================================")
    
    # Find the single ADC file
    test_file = find_adc_file(analysis_dir)
    
    if not test_file:
        print("Error: No valid ADC data files found in analysis directory!")
        print("Please ensure there's a single JX_DEVICEID_YYMMDD.txt file with ADC data in the ./analysis_adc directory.")
        return
    
    # Show which file was selected and extract device ID
    filename = os.path.basename(test_file)
    device_id = extract_device_id_from_filename(filename)
    print(f"Selected file: {filename}")
    
    if device_id:
        print(f"Device ID: {device_id}")
        
        # Extract and show recording date
        jx_pattern = re.compile(r'JX_([A-Z0-9]+)_(\d{6})\.txt')
        match = jx_pattern.match(filename)
        if match:
            recording_date_str = match.group(2)
            try:
                year = 2000 + int(recording_date_str[:2])
                month = int(recording_date_str[2:4])
                day = int(recording_date_str[4:6])
                recording_date = datetime(year, month, day)
                print(f"Recording date: {recording_date.strftime('%Y-%m-%d')}")
            except ValueError:
                pass
    else:
        print("Warning: Could not extract device ID from filename")
    print()
    
    # Check command line arguments for analysis mode
    analysis_mode = len(sys.argv) > 1 and sys.argv[1] in ['--header', '-h', '--analyze', '-a']
    
    try:
        if analysis_mode:
            # Header analysis mode
            print(f"Running comprehensive header analysis on: {test_file}")
            print()
            headers = analyze_adc_headers(test_file)
            print_header_summary(headers)
            
        else:
            # Standard decoding mode
            print(f"Decoding file: {test_file}")
            events = decode_adc_file(test_file)
            
            if not events:
                print("No valid events found in the file.")
                return
            
            # Print summary
            print_event_summary(events)
            
            # Export to CSV
            print(f"\nExporting data to CSV...")
            export_adc_data_to_csv(events)
            
            # Generate plots (if available)
            if PLOTTING_AVAILABLE:
                print(f"\nGenerating plots...")
                plot_event_data(events, device_id)
                print(f"Plots saved to ./analysis_adc/figures/ directory")
            else:
                print(f"\nSkipping plot generation (matplotlib/numpy not available)")
                print("Install with: pip install matplotlib numpy")
            
            print("\nDecoding complete!")
            
            # Show usage hint
            print("\n💡 Tip: Run with --header or -h for comprehensive header analysis")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
