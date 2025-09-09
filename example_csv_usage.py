#!/usr/bin/env python3
"""
Example script showing how to work with the JSON-encoded CSV data from decoder_social.py

This demonstrates how easy it is to parse the device data from the CSV files.
"""

import csv
import json

def read_social_events_csv(csv_path):
    """
    Read and parse the social events CSV with JSON-encoded device data
    
    Args:
        csv_path: Path to social_events.csv
        
    Returns:
        list: List of parsed records
    """
    records = []
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Parse JSON-encoded device data
            device_macs = json.loads(row['device_macs']) if row['device_macs'] else []
            device_rssis = json.loads(row['device_rssis']) if row['device_rssis'] else []
            
            # Create device pairs
            devices = []
            for mac, rssi in zip(device_macs, device_rssis):
                devices.append({'mac': mac, 'rssi': rssi})
            
            record = {
                'timestamp': row['timestamp'],
                'minute_of_day': int(row['minute_of_day']),
                'time': row['time'],
                'record_type': row['record_type'],
                'device_count': int(row['device_count']),
                'motion_count': int(row['motion_count']),
                'battery_level': int(row['battery_level']),
                'temperature_c': int(row['temperature_c']),
                'devices': devices
            }
            
            records.append(record)
    
    return records

def analyze_device_interactions(records):
    """
    Analyze device interactions from the parsed CSV data
    
    Args:
        records: List of parsed records
    """
    print("=== Device Interaction Analysis ===")
    
    # Count device appearances
    device_counts = {}
    device_rssi_ranges = {}
    
    for record in records:
        if record['record_type'] == 'device_scan':
            for device in record['devices']:
                mac = device['mac']
                rssi = device['rssi']
                
                if mac not in device_counts:
                    device_counts[mac] = 0
                    device_rssi_ranges[mac] = {'min': rssi, 'max': rssi}
                
                device_counts[mac] += 1
                device_rssi_ranges[mac]['min'] = min(device_rssi_ranges[mac]['min'], rssi)
                device_rssi_ranges[mac]['max'] = max(device_rssi_ranges[mac]['max'], rssi)
    
    print("Device Detection Counts:")
    for mac, count in sorted(device_counts.items()):
        rssi_range = device_rssi_ranges[mac]
        print(f"  {mac}: {count} detections, RSSI range: {rssi_range['min']} to {rssi_range['max']} dBm")
    
    # Find strongest signals
    print("\nStrongest Signal Detections:")
    strong_signals = []
    for record in records:
        if record['record_type'] == 'device_scan':
            for device in record['devices']:
                if device['rssi'] > -70:  # Strong signal threshold
                    strong_signals.append({
                        'time': record['time'],
                        'device': device['mac'],
                        'rssi': device['rssi']
                    })
    
    for signal in sorted(strong_signals, key=lambda x: x['rssi'], reverse=True)[:5]:
        print(f"  {signal['time']}: {signal['device']} at {signal['rssi']} dBm")

def main():
    """
    Main function to demonstrate CSV usage
    """
    csv_path = './analysis_social/social_events.csv'
    
    try:
        print("Reading social events CSV...")
        records = read_social_events_csv(csv_path)
        print(f"Loaded {len(records)} records")
        
        # Analyze the data
        analyze_device_interactions(records)
        
        # Show sample records
        print("\n=== Sample Records ===")
        device_scan_records = [r for r in records if r['record_type'] == 'device_scan']
        for i, record in enumerate(device_scan_records[:3]):
            print(f"Record {i+1}: {record['time']}")
            print(f"  Devices: {[d['mac'] for d in record['devices']]}")
            print(f"  RSSIs: {[d['rssi'] for d in record['devices']]}")
            print(f"  Motion: {record['motion_count']}, Battery: {record['battery_level']}%")
            print()
        
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run decoder_social.py first to generate the CSV file.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
