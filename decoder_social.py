#!/usr/bin/env python3
"""
Social Data File Format Decoder and Analyzer

This script decodes social interaction data from JUXTA device files and generates plots.
It also provides comprehensive record analysis and validation capabilities.
Based on the specification in spec_Social.md

Follows the same workflow as decode_adc.py but for social interaction data.
"""

import struct
import os
import re
import csv
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

# Optional imports for plotting functionality
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/numpy not available. Plotting functionality disabled.")


def find_social_file(analysis_dir: str) -> Optional[str]:
    """
    Find the single social data file in the directory
    
    Args:
        analysis_dir: Directory to search for .txt files
        
    Returns:
        str: Path to the social data file, or None if no valid file found
    """
    if not os.path.exists(analysis_dir):
        return None
    
    txt_files = [f for f in os.listdir(analysis_dir) if f.endswith('.txt')]
    
    if not txt_files:
        return None
    
    # Pattern to match JX filename format: JX_DEVICEID_YYMMDD.txt
    # Exclude MACIDX files from this search
    jx_pattern = re.compile(r'JX_([A-Z0-9]+)_(\d{6})\.txt')
    
    valid_files = []
    for filename in txt_files:
        # Skip MACIDX files
        if filename.endswith('_MACIDX.txt'):
            continue
            
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
        print("Error: Multiple social data files found in directory:")
        for _, filename, device_id in valid_files:
            print(f"  - {filename} (Device: {device_id})")
        print("Only one social data file is allowed per analysis directory.")
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


def find_macidx_file(analysis_dir: str, device_id: str) -> Optional[str]:
    """
    Find the MACIDX file matching the device ID in the analysis directory
    
    Args:
        analysis_dir: Directory to search for MACIDX files
        device_id: Device ID to match in MACIDX filename
        
    Returns:
        str: Path to the MACIDX file, or None if not found
    """
    if not os.path.exists(analysis_dir) or not device_id:
        return None
    
    txt_files = [f for f in os.listdir(analysis_dir) if f.endswith('.txt')]
    
    # Look for MACIDX files matching the device ID
    expected_macidx_filename = f"JX_{device_id}_MACIDX.txt"
    macidx_files = [f for f in txt_files if f == expected_macidx_filename]
    
    if macidx_files:
        return os.path.join(analysis_dir, macidx_files[0])
    
    # Fallback: look for any MACIDX file (for backwards compatibility)
    fallback_macidx_files = [f for f in txt_files if f.endswith('_MACIDX.txt')]
    if fallback_macidx_files:
        print(f"Warning: Found MACIDX file {fallback_macidx_files[0]} but expected {expected_macidx_filename}")
        return os.path.join(analysis_dir, fallback_macidx_files[0])
    
    return None


def decode_macidx_file(filename: str) -> Dict[int, Dict[str, Any]]:
    """
    Decode MACIDX file to create MAC address lookup table
    
    Args:
        filename: Path to MACIDX file
        
    Returns:
        dict: MAC index -> MAC address mapping
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    mac_table = {}
    index = 0
    
    # Parse 3-byte MAC entries
    for i in range(0, len(file_data), 3):
        if i + 3 <= len(file_data):
            mac_bytes = file_data[i:i+3]
            mac_hex = mac_bytes.hex().upper()
            
            # Create device name - just use the MAC address
            device_name = mac_hex
            
            mac_table[index] = {
                'mac_hex': mac_hex,
                'device_name': device_name,
                'packed_mac': mac_bytes
            }
            index += 1
    
    return mac_table


def convert_minute_to_local_time(minute_of_day: int, use_local_time: bool = True) -> str:
    """
    Convert minute of day to local time string
    
    Args:
        minute_of_day: Minute of day (0-1439)
        use_local_time: If True, use current date with local time. If False, use raw minute format.
        
    Returns:
        str: Formatted time string
    """
    if use_local_time:
        # Use current date and add the minutes
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        local_time = today + timedelta(minutes=minute_of_day)
        return local_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Use simple HH:MM format
        hours = minute_of_day // 60
        minutes = minute_of_day % 60
        return f"{hours:02d}:{minutes:02d}"


def resolve_mac_addresses(records: List[Dict[str, Any]], mac_table: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve MAC indices to actual MAC addresses
    
    Args:
        records: Decoded social records
        mac_table: MAC address lookup table (index -> MAC info)
        
    Returns:
        list: Records with resolved MAC addresses
    """
    for record in records:
        if record['record_type'] == 'device_scan':
            for device in record['devices']:
                mac_index = device['mac_index']
                if mac_index in mac_table:
                    mac_info = mac_table[mac_index]
                    device['mac_address'] = mac_info['mac_hex']
                    device['device_name'] = mac_info['device_name']
                else:
                    device['mac_address'] = f"UNKNOWN_{mac_index}"
                    device['device_name'] = f"UNKNOWN_{mac_index}"
    
    return records


def decode_social_record(file_data: bytes, offset: int = 0, use_local_time: bool = True) -> Optional[Dict[str, Any]]:
    """
    Decode a single social interaction record
    
    Args:
        file_data: Raw file data (bytes)
        offset: Starting offset in file
        
    Returns:
        dict: Decoded record information, or None if invalid
    """
    # Need at least 6 bytes for header
    if len(file_data) < offset + 6:
        return None
        
    header = file_data[offset:offset + 6]
    
    # Unpack header using big-endian format
    minute, device_count, motion_count, battery_level, temperature = struct.unpack('>HBBBB', header)
    
    # Convert temperature from unsigned to signed
    if temperature > 127:
        temperature = temperature - 256
    
    # Determine record type based on device_count
    if device_count >= 0xF1:
        # System event record
        record_type = 'system_event'
        event_names = {
            0xF1: 'boot',
            0xF2: 'connected', 
            0xF3: 'settings',
            0xF5: 'error'
        }
        event_name = event_names.get(device_count, 'unknown')
        record_size = 3  # System events are 3 bytes
        devices = []
    else:
        # Device scan record
        record_size = 6 + (2 * device_count)
        if len(file_data) < offset + record_size:
            return None
        
        # Decode device data if present
        devices = []
        if device_count > 0:
            mac_indices = file_data[offset + 6:offset + 6 + device_count]
            rssi_data = file_data[offset + 6 + device_count:offset + record_size]
            
            for i in range(device_count):
                rssi = rssi_data[i]
                # Convert RSSI from unsigned to signed
                if rssi > 127:
                    rssi = rssi - 256
                    
                devices.append({
                    'mac_index': mac_indices[i],
                    'rssi_dbm': rssi
                })
        
        if device_count == 0:
            record_type = 'no_device_proximity'
            event_name = None
        else:
            record_type = 'device_scan'
            event_name = None
    
    # Convert minute to human time
    time_str = convert_minute_to_local_time(minute, use_local_time)
    
    return {
        'record_type': record_type,
        'event_name': event_name,
        'minute_of_day': minute,
        'time': time_str,
        'device_count': device_count if record_type == 'device_scan' else 0,
        'motion_count': motion_count,
        'battery_level': battery_level,
        'temperature_c': temperature,
        'devices': devices,
        'record_size': record_size,
        'next_offset': offset + record_size
    }


def find_valid_records(file_data: bytes, use_local_time: bool = True) -> List[Dict[str, Any]]:
    """
    Find all valid social records in the file data
    
    Args:
        file_data: Raw file data (bytes)
        
    Returns:
        list: List of valid record information
    """
    valid_records = []
    offset = 0
    
    while offset < len(file_data) - 6:  # Need at least 6 bytes for header
        try:
            record = decode_social_record(file_data, offset, use_local_time)
            if record is None:
                break
            valid_records.append(record)
            offset = record['next_offset']
        except (struct.error, IndexError):
            # Can't parse record at this position, try next byte
            offset += 1
    
    return valid_records


def analyze_social_records(filename: str, mac_table: Optional[Dict[int, Dict[str, Any]]] = None, use_local_time: bool = True) -> List[Dict[str, Any]]:
    """
    Analyze social data records and data sections - comprehensive ground truth checker
    
    Args:
        filename: Path to social data file (hex text format)
        
    Returns:
        list: List of record analysis results
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    print(f"File Analysis: {filename}")
    print(f"Hex data length: {len(hex_data)} characters")
    print(f"Binary data length: {len(file_data)} bytes")
    print("=" * 60)
    
    records = []
    discrepancies = []
    offset = 0
    potential_records = []
    
    # First pass: Find all potential records
    print("SCANNING FOR POTENTIAL RECORDS...")
    while offset < len(file_data) - 6:
        try:
            record = decode_social_record(file_data, offset, use_local_time)
            if record is None:
                offset += 1
                continue
                
            potential_records.append(record)
            offset = record['next_offset']
            
        except (struct.error, IndexError):
            offset += 1
    
    print(f"Found {len(potential_records)} potential records")
    print()
    
    # Second pass: Validate and categorize records
    print("VALIDATING RECORDS...")
    valid_records = []
    invalid_records = []
    
    for i, pr in enumerate(potential_records):
        # Check validation criteria
        issues = []
        
        if pr['minute_of_day'] > 1439:
            issues.append(f"Invalid minute of day: {pr['minute_of_day']} (should be 0-1439)")
        
        if pr['battery_level'] > 100:
            issues.append(f"Invalid battery level: {pr['battery_level']} (should be 0-100)")
        
        if pr['temperature_c'] < -40 or pr['temperature_c'] > 85:
            issues.append(f"Invalid temperature: {pr['temperature_c']}°C (should be -40 to 85)")
        
        if pr['record_type'] == 'device_scan' and pr['device_count'] > 128:
            issues.append(f"Invalid device count: {pr['device_count']} (should be 0-128)")
        
        if issues:
            invalid_records.append({
                'offset': pr.get('next_offset', 0) - pr['record_size'],
                'issues': issues,
                'record_type': pr['record_type']
            })
        else:
            valid_records.append(pr)
    
    # Third pass: Analyze valid records for data completeness and timing
    print("ANALYZING VALID RECORDS...")
    
    # Resolve MAC addresses if MAC table is available (for display purposes)
    if mac_table:
        for vr in valid_records:
            if vr['record_type'] == 'device_scan':
                for device in vr['devices']:
                    mac_index = device['mac_index']
                    if mac_index in mac_table:
                        mac_info = mac_table[mac_index]
                        device['mac_address'] = mac_info['mac_hex']
                        device['device_name'] = mac_info['device_name']
                    else:
                        device['mac_address'] = f"UNKNOWN_{mac_index}"
                        device['device_name'] = f"UNKNOWN_{mac_index}"
    
    for i, vr in enumerate(valid_records):
        # Calculate absolute timestamp (approximate, based on minute of day)
        # Note: This requires a base date - using current date as placeholder
        base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        absolute_timestamp = base_date + timedelta(minutes=vr['minute_of_day'])
        
        record_info = {
            'record_number': i + 1,
            'record_type': vr['record_type'],
            'event_name': vr['event_name'],
            'minute_of_day': vr['minute_of_day'],
            'time': vr['time'],
            'device_count': vr['device_count'],
            'motion_count': vr['motion_count'],
            'battery_level': vr['battery_level'],
            'temperature_c': vr['temperature_c'],
            'devices': vr['devices'],
            'absolute_timestamp': absolute_timestamp,
            'record_size': vr['record_size']
        }
        
        records.append(record_info)
        
        # Check for discrepancies
        if i > 0:
            prev_minute = records[i-1]['minute_of_day']
            minute_diff = vr['minute_of_day'] - prev_minute
            if minute_diff < 0:
                discrepancies.append(f"Record {i+1}: Negative time difference from previous record: {minute_diff} minutes")
            elif minute_diff > 60:  # More than 1 hour gap
                discrepancies.append(f"Record {i+1}: Large time gap from previous record: {minute_diff} minutes")
        
        # Print record analysis
        print(f"Record {record_info['record_number']} ({vr['record_type']}):")
        print(f"  📍 FILE DATA (Raw from file):")
        print(f"    Minute of day: {vr['minute_of_day']} ({vr['time']})")
        print(f"    Device count: {vr['device_count']}")
        print(f"    Motion count: {vr['motion_count']}")
        print(f"    Battery level: {vr['battery_level']}%")
        print(f"    Temperature: {vr['temperature_c']}°C")
        if vr['record_type'] == 'system_event':
            print(f"    Event type: {vr['event_name']}")
        print(f"  🧮 CALCULATED VALUES:")
        print(f"    Human time: {vr['time']} (calculated from minute of day)")
        print(f"    Record size: {vr['record_size']} bytes")
        if vr['devices']:
            print(f"    Devices detected: {len(vr['devices'])}")
            for j, device in enumerate(vr['devices']):
                # Use device_name and mac_address if available (after MAC resolution)
                device_name = device.get('device_name', f"MAC_{device['mac_index']}")
                mac_address = device.get('mac_address', f"Index_{device['mac_index']}")
                print(f"      Device {j+1}: {device_name} ({mac_address}), RSSI {device['rssi_dbm']} dBm")
        print()
    
    # Report invalid records
    if invalid_records:
        print("INVALID RECORDS FOUND:")
        for ir in invalid_records:
            print(f"  Offset {ir['offset']}: {ir['record_type']}")
            for issue in ir['issues']:
                print(f"    - {issue}")
        print()
    
    # Store discrepancies for summary
    if records:
        records[0]['discrepancies'] = discrepancies
    
    return records


def print_record_summary(records: List[Dict[str, Any]]) -> None:
    """
    Print summary of record analysis with discrepancy reporting
    
    Args:
        records: List of record analysis results
    """
    if not records:
        print("No valid records found.")
        return
    
    print("=" * 60)
    print("SOCIAL DATA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📊 FILE ANALYSIS RESULTS:")
    print(f"   Total valid records found: {len(records)} (calculated from record validation)")
    
    # Count by record type
    record_type_counts = {}
    for record in records:
        record_type = record['record_type']
        record_type_counts[record_type] = record_type_counts.get(record_type, 0) + 1
    
    print(f"   Record type breakdown:")
    for record_type, count in record_type_counts.items():
        # Use more explicit labels for clarity
        display_name = {
            'no_device_proximity': 'No Device Proximity (Solo Activity)',
            'device_scan': 'Device Scan (Social Interaction)',
            'system_event': 'System Event'
        }.get(record_type, record_type.replace('_', ' ').title())
        print(f"     {display_name}: {count}")
    
    # Device scan analysis
    device_scan_records = [r for r in records if r['record_type'] == 'device_scan']
    if device_scan_records:
        total_devices = sum(r['device_count'] for r in device_scan_records)
        print(f"   Total device detections: {total_devices}")
        
        # Motion analysis
        total_motion = sum(r['motion_count'] for r in records)
        print(f"   Total motion events: {total_motion}")
    
    # Battery and temperature analysis
    if records:
        battery_levels = [r['battery_level'] for r in records]
        temperatures = [r['temperature_c'] for r in records]
        print(f"   Battery range: {min(battery_levels)}% - {max(battery_levels)}%")
        print(f"   Temperature range: {min(temperatures)}°C - {max(temperatures)}°C")
    
    # Time range analysis
    if len(records) > 1:
        first_minute = records[0]['minute_of_day']
        last_minute = records[-1]['minute_of_day']
        time_span_minutes = last_minute - first_minute
        print(f"\n📊 TIMING ANALYSIS:")
        print(f"   Time span: {time_span_minutes} minutes ({time_span_minutes/60:.1f} hours)")
        print(f"   First record: {records[0]['time']}")
        print(f"   Last record: {records[-1]['time']}")
    
    # Report discrepancies
    discrepancies = records[0].get('discrepancies', [])
    if discrepancies:
        print(f"\n🚨 DISCREPANCIES DETECTED:")
        for discrepancy in discrepancies:
            print(f"   - {discrepancy}")
    else:
        print(f"\n✅ NO DISCREPANCIES: Data appears consistent with specification")


def decode_social_file(filename: str, use_local_time: bool = True) -> List[Dict[str, Any]]:
    """
    Decode entire social data file using robust record detection
    
    Args:
        filename: Path to social data file
        
    Returns:
        list: List of decoded social records
    """
    with open(filename, 'r') as f:
        hex_data = f.read().strip()
    
    # Convert hex string to binary data
    file_data = bytes.fromhex(hex_data)
    
    # Find all valid records
    valid_records = find_valid_records(file_data, use_local_time)
    
    return valid_records


def extract_recording_day_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract the recording day from JX filename format
    
    Args:
        filename: Filename in format JX_DEVICEID_YYMMDD.txt
        
    Returns:
        datetime: Recording day as datetime object, or None if parsing fails
    """
    # Pattern: JX_DEVICEID_YYMMDD.txt
    # We want the YYMMDD part (recording day)
    pattern = re.compile(r'JX_[A-Z0-9]+_(\d{6})\.txt')
    match = pattern.match(os.path.basename(filename))
    
    if match:
        recording_day_str = match.group(1)  # YYMMDD format
        try:
            # Convert YYMMDD to full date (assuming 20XX for years)
            year = 2000 + int(recording_day_str[:2])
            month = int(recording_day_str[2:4])
            day = int(recording_day_str[4:6])
            return datetime(year, month, day)
        except ValueError:
            return None
    
    return None


def convert_minute_of_day_to_timestamps(minute_of_day: int, recording_date: datetime) -> tuple:
    """
    Convert minute of day to both UTC and local timestamps
    
    The device stores minute_of_day in UTC time, so we need to:
    1. Create UTC timestamp from recording_date + minute_of_day  
    2. Convert to local time for analysis
    
    Args:
        minute_of_day: Minute within the day (0-1439) - originally set in UTC
        recording_date: The date when the data was recorded (extracted from filename)
        
    Returns:
        tuple: (utc_timestamp, local_timestamp)
    """
    from datetime import timezone
    
    # The recording_date is extracted from filename (YYMMDD) which represents the UTC date
    # Create UTC time from recording date + minute_of_day
    utc_base_time = recording_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    utc_time = utc_base_time + timedelta(minutes=minute_of_day)
    
    # Convert UTC to local time
    local_time = utc_time.astimezone()
    
    return utc_time.timestamp(), local_time.timestamp()


def convert_minute_of_day_to_unix_timestamp(minute_of_day: int, recording_date: datetime) -> float:
    """
    Convert minute of day to Unix timestamp using the recording date (legacy function)
    Now returns local timestamp for backward compatibility
    
    Args:
        minute_of_day: Minute within the day (0-1439)
        recording_date: The date when the data was recorded
        
    Returns:
        float: Local Unix timestamp
    """
    _, local_timestamp = convert_minute_of_day_to_timestamps(minute_of_day, recording_date)
    return local_timestamp


def calculate_time_axis_range_unix(timestamps: List[float]) -> tuple:
    """
    Calculate appropriate time axis range and ticks based on Unix timestamps
    
    Args:
        timestamps: List of Unix timestamps
        
    Returns:
        tuple: (min_timestamp, max_timestamp, tick_positions, tick_labels)
    """
    if not timestamps:
        return 0, 86400, list(range(0, 86400, 3600)), [f"{h:02d}:00" for h in range(24)]
    
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    data_span_seconds = max_timestamp - min_timestamp
    
    # Add some padding (10% of data span, minimum 30 minutes)
    padding_seconds = max(1800, int(data_span_seconds * 0.1))  # 30 minutes minimum
    plot_min = min_timestamp - padding_seconds
    plot_max = max_timestamp + padding_seconds
    
    # Calculate appropriate tick spacing based on data span
    if data_span_seconds <= 7200:  # 2 hours or less
        tick_interval_seconds = 900  # Every 15 minutes
    elif data_span_seconds <= 21600:  # 6 hours or less
        tick_interval_seconds = 1800  # Every 30 minutes
    elif data_span_seconds <= 43200:  # 12 hours or less
        tick_interval_seconds = 3600  # Every hour
    else:
        tick_interval_seconds = 7200  # Every 2 hours
    
    # Generate tick positions
    first_tick_time = datetime.fromtimestamp(min_timestamp)
    first_tick_rounded = first_tick_time.replace(second=0, microsecond=0)
    
    # Round to nearest tick interval
    minutes_to_round = tick_interval_seconds // 60
    first_tick_rounded = first_tick_rounded.replace(
        minute=(first_tick_rounded.minute // minutes_to_round) * minutes_to_round
    )
    
    tick_positions = []
    tick_labels = []
    current_tick = first_tick_rounded.timestamp()
    
    while current_tick <= plot_max:
        if current_tick >= plot_min:
            tick_positions.append(current_tick)
            tick_time = datetime.fromtimestamp(current_tick)
            if tick_interval_seconds < 3600:  # Show minutes for sub-hour intervals
                tick_labels.append(tick_time.strftime('%H:%M'))
            else:
                tick_labels.append(tick_time.strftime('%H:%00'))
        current_tick += tick_interval_seconds
    
    return plot_min, plot_max, tick_positions, tick_labels


def calculate_time_axis_range(minutes: List[int]) -> tuple:
    """
    Calculate appropriate time axis range and ticks based on minute values (legacy)
    
    Args:
        minutes: List of minute values from records
        
    Returns:
        tuple: (min_minute, max_minute, tick_positions, tick_labels)
    """
    if not minutes:
        return 0, 1439, list(range(0, 1440, 60)), [f"{h:02d}:00" for h in range(24)]
    
    min_minute = min(minutes)
    max_minute = max(minutes)
    data_span = max_minute - min_minute
    
    # Add some padding (10% of data span, minimum 30 minutes)
    padding = max(30, int(data_span * 0.1))
    plot_min = max(0, min_minute - padding)
    plot_max = min(1439, max_minute + padding)
    
    # Calculate appropriate tick spacing based on data span
    if data_span <= 120:  # 2 hours or less
        tick_interval = 15  # Every 15 minutes
    elif data_span <= 360:  # 6 hours or less
        tick_interval = 30  # Every 30 minutes
    elif data_span <= 720:  # 12 hours or less
        tick_interval = 60  # Every hour
    else:
        tick_interval = 120  # Every 2 hours
    
    # Generate ticks within the plot range
    first_tick = (plot_min // tick_interval) * tick_interval
    tick_positions = list(range(first_tick, plot_max + tick_interval, tick_interval))
    
    # Generate labels
    tick_labels = []
    for tick in tick_positions:
        hours = tick // 60
        mins = tick % 60
        if tick_interval < 60:
            tick_labels.append(f"{hours:02d}:{mins:02d}")
        else:
            tick_labels.append(f"{hours:02d}:00")
    
    return plot_min, plot_max, tick_positions, tick_labels


def plot_social_data(records: List[Dict[str, Any]], filename: str = "", device_id: str = "", output_dir: str = "./analysis_social/figures") -> None:
    """
    Generate plots for social interaction data
    
    Args:
        records: List of decoded social records
        filename: Source filename (used to extract recording date for proper timestamps)
        device_id: Device ID for plot titles
        output_dir: Directory to save plot files
    """
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting functionality requires matplotlib and numpy.")
        print("Please install with: pip install matplotlib numpy")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract recording date from filename for proper timestamp conversion
    recording_date = None
    use_unix_timestamps = False
    
    if filename:
        recording_date = extract_recording_day_from_filename(filename)
        if recording_date:
            use_unix_timestamps = True
            print(f"Using recording date {recording_date.strftime('%Y-%m-%d')} from filename for timestamp conversion")
        else:
            print("Could not extract recording date from filename, using minute-of-day values")
    
    # Clear existing files in the figures directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted existing file: {filename}")
    
    # Separate records by type
    device_scan_records = [r for r in records if r['record_type'] == 'device_scan']
    system_event_records = [r for r in records if r['record_type'] == 'system_event']
    no_device_proximity_records = [r for r in records if r['record_type'] == 'no_device_proximity']
    
    # Calculate global time range once for consistent axis limits across all plots
    global_plot_min = None
    global_plot_max = None
    global_tick_positions = None
    global_tick_labels = None
    
    if records:
        if use_unix_timestamps and recording_date:
            # Calculate global range using Unix timestamps
            all_timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            global_plot_min, global_plot_max, global_tick_positions, global_tick_labels = calculate_time_axis_range_unix(all_timestamps)
        else:
            # Calculate global range using minute_of_day values
            all_minutes = [r['minute_of_day'] for r in records]
            global_plot_min, global_plot_max, global_tick_positions, global_tick_labels = calculate_time_axis_range(all_minutes)
        
        print(f"Global plot range: {global_plot_min} to {global_plot_max} ({len(global_tick_positions)} ticks)")
    
    # Plot 1: Device detection timeline
    if device_scan_records or no_device_proximity_records:
        plt.figure(figsize=(15, 8))
        
        if use_unix_timestamps and recording_date:
            # Convert minute_of_day to Unix timestamps
            timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            device_counts = [r['device_count'] for r in records]
            
            plt.scatter(timestamps, device_counts, c='blue', alpha=0.7, s=50, edgecolors='darkblue', linewidth=0.5)
            
            # Use global axis range for consistency
            plt.xlim(global_plot_min, global_plot_max)
            plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
            
            plt.xlabel('Local Time')
            device_title = f" (Device: {device_id})" if device_id else ""
            plt.title(f'Social Interaction Timeline - Device Detections{device_title} ({recording_date.strftime("%Y-%m-%d")})')
        else:
            # Fallback to minute_of_day values
            minutes = [r['minute_of_day'] for r in records]
            device_counts = [r['device_count'] for r in records]
            
            plt.scatter(minutes, device_counts, c='blue', alpha=0.7, s=50, edgecolors='darkblue', linewidth=0.5)
            
            # Use global axis range for consistency
            plt.xlim(global_plot_min, global_plot_max)
            plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
            
            plt.xlabel('Time of Day')
            device_title = f" (Device: {device_id})" if device_id else ""
            plt.title(f'Social Interaction Timeline - Device Detections{device_title}')
        
        plt.ylabel('Number of Devices Detected')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'device_timeline.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Motion detection timeline
    if records:
        plt.figure(figsize=(15, 6))
        
        if use_unix_timestamps and recording_date:
            # Convert minute_of_day to Unix timestamps
            timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            motion_counts = [r['motion_count'] for r in records]
            
            plt.scatter(timestamps, motion_counts, c='green', alpha=0.7, s=50, edgecolors='darkgreen', linewidth=0.5)
            plt.xlabel('Local Time')
            device_title = f" (Device: {device_id})" if device_id else ""
            plt.title(f'Motion Detection Timeline{device_title} ({recording_date.strftime("%Y-%m-%d")})')
        else:
            # Fallback to minute_of_day values
            minutes = [r['minute_of_day'] for r in records]
            motion_counts = [r['motion_count'] for r in records]
            
            plt.scatter(minutes, motion_counts, c='green', alpha=0.7, s=50, edgecolors='darkgreen', linewidth=0.5)
            plt.xlabel('Time of Day')
            device_title = f" (Device: {device_id})" if device_id else ""
            plt.title(f'Motion Detection Timeline{device_title}')
        
        plt.ylabel('Motion Events per Minute')
        plt.grid(True, alpha=0.3)
        
        # Use global axis range for consistency
        plt.xlim(global_plot_min, global_plot_max)
        plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'motion_timeline.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Battery and temperature monitoring
    if records:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        if use_unix_timestamps and recording_date:
            # Convert minute_of_day to Unix timestamps
            timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            battery_levels = [r['battery_level'] for r in records]
            temperatures = [r['temperature_c'] for r in records]
            
            # Battery plot
            ax1.scatter(timestamps, battery_levels, c='red', alpha=0.7, s=50, edgecolors='darkred', linewidth=0.5)
            ax1.set_xlabel('Local Time')
            ax1.set_ylabel('Battery Level (%)')
            ax1.set_title(f'Battery Level Over Time ({recording_date.strftime("%Y-%m-%d")})')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Temperature plot
            ax2.scatter(timestamps, temperatures, c='orange', alpha=0.7, s=50, edgecolors='darkorange', linewidth=0.5)
            ax2.set_xlabel('Local Time')
            ax2.set_ylabel('Temperature (°C)')
            ax2.set_title(f'Temperature Over Time ({recording_date.strftime("%Y-%m-%d")})')
            ax2.grid(True, alpha=0.3)
            
            # Use global axis range for consistency
            ax1.set_xlim(global_plot_min, global_plot_max)
            ax2.set_xlim(global_plot_min, global_plot_max)
            ax1.set_xticks(global_tick_positions)
            ax1.set_xticklabels(global_tick_labels, rotation=45)
            ax2.set_xticks(global_tick_positions)
            ax2.set_xticklabels(global_tick_labels, rotation=45)
        else:
            # Fallback to minute_of_day values
            minutes = [r['minute_of_day'] for r in records]
            battery_levels = [r['battery_level'] for r in records]
            temperatures = [r['temperature_c'] for r in records]
            
            # Battery plot
            ax1.scatter(minutes, battery_levels, c='red', alpha=0.7, s=50, edgecolors='darkred', linewidth=0.5)
            ax1.set_xlabel('Time of Day')
            ax1.set_ylabel('Battery Level (%)')
            ax1.set_title('Battery Level Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Temperature plot
            ax2.scatter(minutes, temperatures, c='orange', alpha=0.7, s=50, edgecolors='darkorange', linewidth=0.5)
            ax2.set_xlabel('Time of Day')
            ax2.set_ylabel('Temperature (°C)')
            ax2.set_title('Temperature Over Time')
            ax2.grid(True, alpha=0.3)
            
            # Use global axis range for consistency
            ax1.set_xlim(global_plot_min, global_plot_max)
            ax2.set_xlim(global_plot_min, global_plot_max)
            ax1.set_xticks(global_tick_positions)
            ax1.set_xticklabels(global_tick_labels, rotation=45)
            ax2.set_xticks(global_tick_positions)
            ax2.set_xticklabels(global_tick_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'battery_temperature.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: RSSI Timeline - Device proximity over time
    if device_scan_records:
        plt.figure(figsize=(15, 8))
        
        # Collect all device data with timestamps
        device_data = {}
        for record in device_scan_records:
            if use_unix_timestamps and recording_date:
                timestamp = convert_minute_of_day_to_unix_timestamp(record['minute_of_day'], recording_date)
            else:
                timestamp = record['minute_of_day']
            
            for device in record['devices']:
                device_name = device.get('device_name', f"MAC_{device['mac_index']}")
                if device_name not in device_data:
                    device_data[device_name] = {'timestamps': [], 'rssi': []}
                device_data[device_name]['timestamps'].append(timestamp)
                device_data[device_name]['rssi'].append(device['rssi_dbm'])
        
        # Plot each device as a separate scatter plot
        colors = plt.cm.tab10(np.linspace(0, 1, len(device_data)))
        for i, (device_name, data) in enumerate(device_data.items()):
            plt.scatter(data['timestamps'], data['rssi'], 
                       c=[colors[i]], s=60, alpha=0.8, 
                       label=device_name, edgecolors='black', linewidth=0.5)
        
        if use_unix_timestamps and recording_date:
            plt.xlabel('Local Time')
            plt.title(f'Device Proximity Timeline - RSSI Signal Strength Over Time ({recording_date.strftime("%Y-%m-%d")})')
        else:
            plt.xlabel('Time of Day')
            plt.title('Device Proximity Timeline - RSSI Signal Strength Over Time')
        
        plt.ylabel('RSSI (dBm)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Use global axis range for consistency
        plt.xlim(global_plot_min, global_plot_max)
        plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
        
        # Add RSSI interpretation guide
        plt.axhline(y=-30, color='green', linestyle='--', alpha=0.5, label='Very Close (~-30 dBm)')
        plt.axhline(y=-60, color='orange', linestyle='--', alpha=0.5, label='Close (~-60 dBm)')
        plt.axhline(y=-90, color='red', linestyle='--', alpha=0.5, label='Far (~-90 dBm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rssi_timeline.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 5: Data analysis summary
    if len(records) > 1:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Record type distribution over time
        plt.subplot(2, 2, 1)
        
        if use_unix_timestamps and recording_date:
            timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            record_types = [r['record_type'] for r in records]
            
            # Create line plot showing record type changes over time
            type_mapping = {'device_scan': 1, 'system_event': 2, 'no_device_proximity': 0}
            type_values = [type_mapping.get(rt, 0) for rt in record_types]
            
            plt.scatter(timestamps, type_values, c='blue', alpha=0.7, s=40, edgecolors='darkblue', linewidth=0.5)
            plt.xlabel('Local Time')
            plt.title(f'Record Type Over Time ({recording_date.strftime("%Y-%m-%d")})')
        else:
            minutes = [r['minute_of_day'] for r in records]
            record_types = [r['record_type'] for r in records]
            
            # Create line plot showing record type changes over time
            type_mapping = {'device_scan': 1, 'system_event': 2, 'no_device_proximity': 0}
            type_values = [type_mapping.get(rt, 0) for rt in record_types]
            
            plt.scatter(minutes, type_values, c='blue', alpha=0.7, s=40, edgecolors='darkblue', linewidth=0.5)
            plt.xlabel('Time of Day')
            plt.title('Record Type Over Time')
        
        plt.ylabel('Record Type')
        plt.yticks([0, 1, 2], ['No Device Proximity', 'Device Scan', 'System Event'])
        plt.grid(True, alpha=0.3)
        
        # Use global axis range for consistency
        plt.xlim(global_plot_min, global_plot_max)
        plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
        
        # Subplot 2: Device count over time
        plt.subplot(2, 2, 2)
        if device_scan_records:
            if use_unix_timestamps and recording_date:
                timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in device_scan_records]
                device_counts = [r['device_count'] for r in device_scan_records]
                plt.scatter(timestamps, device_counts, c='green', alpha=0.7, s=40, edgecolors='darkgreen', linewidth=0.5)
                plt.xlabel('Local Time')
                plt.title(f'Device Count Over Time ({recording_date.strftime("%Y-%m-%d")})')
            else:
                minutes = [r['minute_of_day'] for r in device_scan_records]
                device_counts = [r['device_count'] for r in device_scan_records]
                plt.scatter(minutes, device_counts, c='green', alpha=0.7, s=40, edgecolors='darkgreen', linewidth=0.5)
                plt.xlabel('Time of Day')
                plt.title('Device Count Over Time')
            
            plt.ylabel('Number of Devices')
            plt.grid(True, alpha=0.3)
            
            # Use global axis range for consistency
            plt.xlim(global_plot_min, global_plot_max)
            plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
        
        # Subplot 3: Motion count over time
        plt.subplot(2, 2, 3)
        
        if use_unix_timestamps and recording_date:
            timestamps = [convert_minute_of_day_to_unix_timestamp(r['minute_of_day'], recording_date) for r in records]
            motion_counts = [r['motion_count'] for r in records]
            plt.scatter(timestamps, motion_counts, c='orange', alpha=0.7, s=40, edgecolors='darkorange', linewidth=0.5)
            plt.xlabel('Local Time')
            plt.title(f'Motion Count Over Time ({recording_date.strftime("%Y-%m-%d")})')
        else:
            minutes = [r['minute_of_day'] for r in records]
            motion_counts = [r['motion_count'] for r in records]
            plt.scatter(minutes, motion_counts, c='orange', alpha=0.7, s=40, edgecolors='darkorange', linewidth=0.5)
            plt.xlabel('Time of Day')
            plt.title('Motion Count Over Time')
        
        plt.ylabel('Motion Events per Minute')
        plt.grid(True, alpha=0.3)
        
        # Use global axis range for consistency
        plt.xlim(global_plot_min, global_plot_max)
        plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
        
        # Subplot 4: Average RSSI over time (if device data available)
        plt.subplot(2, 2, 4)
        if device_scan_records:
            timestamps = []
            avg_rssi = []
            for record in device_scan_records:
                if record['devices']:
                    if use_unix_timestamps and recording_date:
                        timestamp = convert_minute_of_day_to_unix_timestamp(record['minute_of_day'], recording_date)
                    else:
                        timestamp = record['minute_of_day']
                    timestamps.append(timestamp)
                    rssi_values = [device['rssi_dbm'] for device in record['devices']]
                    avg_rssi.append(sum(rssi_values) / len(rssi_values))
            
            if avg_rssi:
                plt.scatter(timestamps, avg_rssi, c='purple', alpha=0.7, s=40, edgecolors='indigo', linewidth=0.5)
                
                if use_unix_timestamps and recording_date:
                    plt.xlabel('Local Time')
                    plt.title(f'Average RSSI Over Time ({recording_date.strftime("%Y-%m-%d")})')
                else:
                    plt.xlabel('Time of Day')
                    plt.title('Average RSSI Over Time')
                
                plt.ylabel('Average RSSI (dBm)')
                plt.grid(True, alpha=0.3)
                
                # Use global axis range for consistency
                plt.xlim(global_plot_min, global_plot_max)
                plt.xticks(global_tick_positions, global_tick_labels, rotation=45)
            else:
                plt.text(0.5, 0.5, 'No RSSI data available', ha='center', va='center', transform=plt.gca().transAxes)
                if use_unix_timestamps and recording_date:
                    plt.title(f'Average RSSI Over Time ({recording_date.strftime("%Y-%m-%d")})')
                else:
                    plt.title('Average RSSI Over Time')
        else:
            plt.text(0.5, 0.5, 'No device scan data available', ha='center', va='center', transform=plt.gca().transAxes)
            if use_unix_timestamps and recording_date:
                plt.title(f'Average RSSI Over Time ({recording_date.strftime("%Y-%m-%d")})')
            else:
                plt.title('Average RSSI Over Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'social_summary.jpg'), dpi=300, bbox_inches='tight')
        plt.close()


def export_social_data_to_csv(records: List[Dict[str, Any]], filename: str = "", output_dir: str = "./analysis_social", use_local_time: bool = True) -> None:
    """
    Export social data to CSV files, separated by event type
    
    Args:
        records: List of decoded social records
        filename: Source filename (used to extract recording date for proper timestamps)
        output_dir: Directory to save CSV files
        use_local_time: Whether to use local time format for display strings
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract recording date from filename for proper timestamp conversion
    recording_date = None
    use_timestamp_conversion = False
    
    if filename:
        recording_date = extract_recording_day_from_filename(filename)
        if recording_date:
            use_timestamp_conversion = True
            print(f"CSV export using recording date {recording_date.strftime('%Y-%m-%d')} for UTC/local timestamp conversion")
    
    # Separate records by type
    system_events = [r for r in records if r['record_type'] == 'system_event']
    social_events = [r for r in records if r['record_type'] in ['device_scan', 'no_device_proximity']]
    
    # Export System Events CSV
    if system_events:
        system_csv_path = os.path.join(output_dir, 'system_events.csv')
        with open(system_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['minute_of_day', 'utc_timestamp', 'local_timestamp', 'utc_datetime', 'local_datetime', 
                         'event_type', 'event_name', 'battery_level', 'temperature_c']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in system_events:
                if use_timestamp_conversion and recording_date:
                    # Convert minute_of_day to both UTC and local timestamps
                    utc_timestamp, local_timestamp = convert_minute_of_day_to_timestamps(record['minute_of_day'], recording_date)
                    utc_datetime = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    local_datetime = datetime.fromtimestamp(local_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Fallback to original values
                    utc_timestamp = ''
                    local_timestamp = record.get('absolute_timestamp', '')
                    utc_datetime = ''
                    local_datetime = record['time'] if use_local_time else convert_minute_to_local_time(record['minute_of_day'], False)
                
                writer.writerow({
                    'minute_of_day': record['minute_of_day'],
                    'utc_timestamp': utc_timestamp,
                    'local_timestamp': local_timestamp,
                    'utc_datetime': utc_datetime,
                    'local_datetime': local_datetime,
                    'event_type': record['record_type'],
                    'event_name': record.get('event_name', ''),
                    'battery_level': record['battery_level'],
                    'temperature_c': record['temperature_c']
                })
        print(f"System events exported to: {system_csv_path}")
    
    # Export Social Events CSV
    if social_events:
        social_csv_path = os.path.join(output_dir, 'social_events.csv')
        
        with open(social_csv_path, 'w', newline='') as csvfile:
            # Create fieldnames with JSON-encoded device columns and UTC/local timestamps
            fieldnames = ['minute_of_day', 'utc_timestamp', 'local_timestamp', 'utc_datetime', 'local_datetime',
                         'record_type', 'device_count', 'motion_count', 'battery_level', 'temperature_c', 
                         'device_macs', 'device_rssis']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in social_events:
                # Extract device MACs and RSSIs
                device_macs = []
                device_rssis = []
                
                for device in record.get('devices', []):
                    device_macs.append(device.get('device_name', f"MAC_{device['mac_index']}"))
                    device_rssis.append(device['rssi_dbm'])
                
                if use_timestamp_conversion and recording_date:
                    # Convert minute_of_day to both UTC and local timestamps
                    utc_timestamp, local_timestamp = convert_minute_of_day_to_timestamps(record['minute_of_day'], recording_date)
                    utc_datetime = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    local_datetime = datetime.fromtimestamp(local_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    # Fallback to original values
                    utc_timestamp = ''
                    local_timestamp = record.get('absolute_timestamp', '')
                    utc_datetime = ''
                    local_datetime = record['time'] if use_local_time else convert_minute_to_local_time(record['minute_of_day'], False)
                
                row_data = {
                    'minute_of_day': record['minute_of_day'],
                    'utc_timestamp': utc_timestamp,
                    'local_timestamp': local_timestamp,
                    'utc_datetime': utc_datetime,
                    'local_datetime': local_datetime,
                    'record_type': record['record_type'],
                    'device_count': record['device_count'],
                    'motion_count': record['motion_count'],
                    'battery_level': record['battery_level'],
                    'temperature_c': record['temperature_c'],
                    'device_macs': json.dumps(device_macs) if device_macs else '',
                    'device_rssis': json.dumps(device_rssis) if device_rssis else ''
                }
                
                writer.writerow(row_data)
        print(f"Social events exported to: {social_csv_path}")


def print_social_summary(records: List[Dict[str, Any]]) -> None:
    """
    Print a summary of decoded social data
    
    Args:
        records: List of decoded social records
    """
    print(f"\n=== Social Data Processing Summary ===")
    print(f"Total records found: {len(records)}")
    
    # Count by record type
    record_type_counts = {}
    for record in records:
        record_type = record['record_type']
        record_type_counts[record_type] = record_type_counts.get(record_type, 0) + 1
    
    print(f"Record type breakdown:")
    for record_type, count in record_type_counts.items():
        # Use more explicit labels for clarity
        display_name = {
            'no_device_proximity': 'No Device Proximity (Solo Activity)',
            'device_scan': 'Device Scan (Social Interaction)',
            'system_event': 'System Event'
        }.get(record_type, record_type.replace('_', ' ').title())
        print(f"  {display_name}: {count}")
    
    # Count total device detections
    device_scan_records = [r for r in records if r['record_type'] == 'device_scan']
    total_devices = sum(r['device_count'] for r in device_scan_records)
    print(f"Total device detections: {total_devices}")
    
    # Count total motion events
    total_motion = sum(r['motion_count'] for r in records)
    print(f"Total motion events: {total_motion}")
    
    if records:
        print(f"\nFirst record time: {records[0]['time']}")
        print(f"Last record time: {records[-1]['time']}")
        
        print(f"\nRecord Details:")
        for i, record in enumerate(records[:10]):  # Show first 10 records
            record_type_display = record['record_type'].replace('_', ' ').title()
            print(f"  Record {i+1} ({record_type_display}): {record['time']}")
            print(f"    Devices: {record['device_count']}, Motion: {record['motion_count']}")
            print(f"    Battery: {record['battery_level']}%, Temperature: {record['temperature_c']}°C")
            if record['devices']:
                for j, device in enumerate(record['devices']):
                    device_name = device.get('device_name', f"MAC_{device['mac_index']}")
                    mac_address = device.get('mac_address', f"Index_{device['mac_index']}")
                    print(f"      Device {j+1}: {device_name} ({mac_address}), RSSI {device['rssi_dbm']} dBm")
            print()
        
        if len(records) > 10:
            print(f"  ... and {len(records) - 10} more records")


def main():
    """
    Main function to decode social data file and generate plots, with optional record analysis
    """
    import sys
    
    # Look for the most recent juxta file in the analysis directory
    analysis_dir = './analysis_social'
    if not os.path.exists(analysis_dir):
        print(f"Analysis directory '{analysis_dir}' not found!")
        return
    
    print("Social Data Decoder and Record Analyzer")
    print("======================================")
    
    # Find the single social data file
    test_file = find_social_file(analysis_dir)
    
    if not test_file:
        print("Error: No valid social data files found in analysis directory!")
        print("Please ensure there's a single JX_DEVICEID_YYMMDD.txt file with social data in the ./analysis_social directory.")
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
    
    # Check command line arguments for analysis mode and time format
    analysis_mode = len(sys.argv) > 1 and sys.argv[1] in ['--header', '-h', '--analyze', '-a']
    use_simple_time = '--simple-time' in sys.argv or '-s' in sys.argv
    use_local_time = not use_simple_time  # Default to local time
    
    # Look for MACIDX file for MAC address resolution
    macidx_file = None
    mac_table = None
    
    if device_id:
        macidx_file = find_macidx_file(analysis_dir, device_id)
        
        if macidx_file:
            print(f"Found MACIDX file: {os.path.basename(macidx_file)}")
            try:
                mac_table = decode_macidx_file(macidx_file)
                print(f"Loaded {len(mac_table)} MAC addresses from MACIDX file")
                for index, mac_info in mac_table.items():
                    print(f"  Index {index}: {mac_info['device_name']} ({mac_info['mac_hex']})")
            except Exception as e:
                print(f"Warning: Failed to decode MACIDX file: {e}")
                mac_table = None
        else:
            print(f"No MACIDX file found (expected: JX_{device_id}_MACIDX.txt)")
            print("Device names will show as MAC indices only")
    else:
        print("No device ID available - cannot look for MACIDX file")
        print("Device names will show as MAC indices only")
    print()
    
    try:
        if analysis_mode:
            # Record analysis mode
            print(f"Running comprehensive record analysis on: {test_file}")
            print()
            records = analyze_social_records(test_file, mac_table, use_local_time=use_local_time)
            
            # Resolve MAC addresses if MAC table is available
            if mac_table:
                records = resolve_mac_addresses(records, mac_table)
                print("MAC addresses resolved using MACIDX file")
            
            print_record_summary(records)
            
        else:
            # Standard decoding mode
            print(f"Decoding file: {test_file}")
            records = decode_social_file(test_file, use_local_time=use_local_time)
            
            if not records:
                print("No valid records found in the file.")
                return
            
            # Resolve MAC addresses if MAC table is available
            if mac_table:
                records = resolve_mac_addresses(records, mac_table)
                print("MAC addresses resolved using MACIDX file")
            
            # Check if we have RSSI data and warn about missing MACIDX if needed
            device_scan_records = [r for r in records if r['record_type'] == 'device_scan']
            if device_scan_records and not mac_table:
                print("Warning: RSSI data detected but no MACIDX file found!")
                print("Device names will show as indices instead of readable names.")
                print(f"Expected MACIDX file: JX_{device_id}_MACIDX.txt")
            
            # Print summary
            print_social_summary(records)
            
            # Export to CSV
            print(f"\nExporting data to CSV...")
            export_social_data_to_csv(records, test_file, use_local_time=use_local_time)
            
            # Generate plots (if available)
            if PLOTTING_AVAILABLE:
                print(f"\nGenerating plots...")
                plot_social_data(records, test_file, device_id)
                print(f"Plots saved to ./analysis_social/figures/ directory")
            else:
                print(f"\nSkipping plot generation (matplotlib/numpy not available)")
                print("Install with: pip install matplotlib numpy")
            
            print("\nDecoding complete!")
            
            # Show usage hint
            print("\n💡 Tip: Run with --header or -h for comprehensive record analysis")
            print("💡 Tip: Run with --simple-time or -s to use HH:MM format instead of local timestamps")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
