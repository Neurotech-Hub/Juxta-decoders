#!/usr/bin/env python3
"""
ADC Header Analyzer

Simplified script to extract and analyze ADC burst headers and data sections.
This script focuses on header extraction and data size analysis.
"""

import struct
import os
from datetime import datetime
from typing import List, Dict, Any


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
    
    # First pass: Find all potential headers (every 12-byte alignment)
    print("SCANNING FOR POTENTIAL HEADERS...")
    while offset < len(file_data) - 12:
        try:
            header = file_data[offset:offset + 12]
            unix_timestamp, microsecond_offset, sample_count, duration_us = struct.unpack('>IIHH', header)
            
            potential_headers.append({
                'offset': offset,
                'unix_timestamp': unix_timestamp,
                'microsecond_offset': microsecond_offset,
                'sample_count': sample_count,
                'duration_us': duration_us,
                'raw_header': header.hex()
            })
            
            offset += 12  # Check every 12-byte boundary
            
        except (struct.error, IndexError):
            offset += 12
    
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
        
        if ph['unix_timestamp'] < 1000000000 or ph['unix_timestamp'] > 2000000000:
            issues.append(f"Invalid timestamp: {ph['unix_timestamp']} (should be 2001-2033)")
        
        # Check if this looks like sample data (all same value)
        if ph['sample_count'] > 0 and ph['sample_count'] < 10000:
            data_start = ph['offset'] + 12
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
        data_start = vh['offset'] + 12
        data_end = data_start + vh['sample_count']
        actual_data_length = min(vh['sample_count'], len(file_data) - data_start)
        
        # Calculate absolute timestamp
        absolute_timestamp = vh['unix_timestamp'] + (vh['microsecond_offset'] / 1_000_000.0)
        
        header_info = {
            'burst_number': i + 1,
            'offset': vh['offset'],
            'unix_timestamp': vh['unix_timestamp'],
            'microsecond_offset': vh['microsecond_offset'],
            'sample_count': vh['sample_count'],
            'duration_us': vh['duration_us'],
            'absolute_timestamp': absolute_timestamp,
            'data_start': data_start,
            'data_end': data_end,
            'expected_data_length': vh['sample_count'],
            'actual_data_length': actual_data_length,
            'data_complete': actual_data_length == vh['sample_count'],
            'next_header_offset': data_end
        }
        
        headers.append(header_info)
        
        # Check for discrepancies
        if not header_info['data_complete']:
            discrepancies.append(f"Burst {i+1}: Incomplete data - {actual_data_length}/{vh['sample_count']} bytes")
        
        # Check timing consistency
        if i > 0:
            prev_timestamp = headers[i-1]['absolute_timestamp']
            time_diff = absolute_timestamp - prev_timestamp
            if time_diff < 0:
                discrepancies.append(f"Burst {i+1}: Negative time difference from previous burst: {time_diff:.3f}s")
            elif time_diff > 3600:  # More than 1 hour
                discrepancies.append(f"Burst {i+1}: Large time gap from previous burst: {time_diff:.1f}s")
        
        # Print header analysis
        print(f"Burst {header_info['burst_number']}:")
        print(f"  📍 FILE DATA (Raw from file):")
        print(f"    Header offset: {vh['offset']} bytes")
        print(f"    Raw header hex: {file_data[vh['offset']:vh['offset']+12].hex()}")
        print(f"    Unix timestamp: {vh['unix_timestamp']} (raw 32-bit value)")
        print(f"    Microsecond offset: {vh['microsecond_offset']} (raw 32-bit value)")
        print(f"    Sample count: {vh['sample_count']} (raw 16-bit value)")
        print(f"    Duration: {vh['duration_us']} μs (raw 16-bit value)")
        print(f"  🧮 CALCULATED VALUES:")
        print(f"    Human timestamp: {datetime.fromtimestamp(vh['unix_timestamp'])} (calculated from Unix timestamp)")
        print(f"    Absolute timestamp: {absolute_timestamp:.6f} (Unix + microsecond_offset/1M)")
        print(f"    Data section: bytes {data_start}-{data_end-1} (calculated: offset+12 to offset+12+sample_count)")
        print(f"    Expected data length: {vh['sample_count']} bytes (from sample_count field)")
        print(f"    Actual data available: {actual_data_length} bytes (calculated: min(sample_count, file_size-data_start))")
        print(f"    Data complete: {'YES' if header_info['data_complete'] else 'NO'} (calculated: actual==expected)")
        print(f"    Next header at: {data_end} bytes (calculated: data_start + sample_count)")
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


def print_summary(headers: List[Dict[str, Any]]) -> None:
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


def main():
    """
    Main function to analyze ADC headers
    """
    # Find the only .txt file (excluding requirements.txt)
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
    
    if not txt_files:
        print("No ADC data files found!")
        return
    
    if len(txt_files) > 1:
        print(f"Multiple .txt files found: {txt_files}")
        print("Using the first one...")
    
    filename = txt_files[0]
    print(f"Analyzing file: {filename}")
    print()
    
    try:
        headers = analyze_adc_headers(filename)
        print_summary(headers)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
